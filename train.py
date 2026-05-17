import argparse
import math
import os
import random
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import build_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[:, self.pad_idx] = 0
            true_dist[target.eq(self.pad_idx)] = 0
        loss = -(true_dist * log_probs).sum(dim=1)
        return loss[~target.eq(self.pad_idx)].mean()


def _step_batch(model, src, tgt, loss_fn, device):
    src, tgt = src.to(device), tgt.to(device)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    src_mask = make_src_mask(src, PAD_IDX)
    tgt_mask = make_tgt_mask(tgt_in, PAD_IDX)
    logits = model(src, tgt_in, src_mask, tgt_mask)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

    pred = logits.argmax(dim=-1)
    non_pad = tgt_out.ne(PAD_IDX)
    correct = (pred.eq(tgt_out) & non_pad).sum().item()
    total = non_pad.sum().item()

    probs = torch.softmax(logits, dim=-1)
    correct_token_probs = probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
    confidence_sum = (correct_token_probs * non_pad).sum().item()
    return loss, correct, total, confidence_sum


def _l2_norm_from_grad_tensors(grads):
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += grad.detach().pow(2).sum().item()
    return total ** 0.5


def get_attention_projection_grad_norms(model: Transformer):
    query_grads = []
    key_grads = []
    metrics = {}

    for name, module in model.named_modules():
        if hasattr(module, 'w_q') and hasattr(module, 'w_k'):
            q_grad = module.w_q.weight.grad
            k_grad = module.w_k.weight.grad
            query_grads.append(q_grad)
            key_grads.append(k_grad)
            safe_name = name.replace('.', '/')
            metrics[f'grad_norm/query/{safe_name}'] = _l2_norm_from_grad_tensors([q_grad])
            metrics[f'grad_norm/key/{safe_name}'] = _l2_norm_from_grad_tensors([k_grad])

    metrics['grad_norm/query/all_attention'] = _l2_norm_from_grad_tensors(query_grads)
    metrics['grad_norm/key/all_attention'] = _l2_norm_from_grad_tensors(key_grads)
    return metrics


def _tokens_from_tensor(token_ids: torch.Tensor, vocab):
    tokens = []
    for idx in token_ids.detach().cpu().tolist():
        if idx == PAD_IDX:
            continue
        tokens.append(vocab.lookup_token(idx))
    return tokens


def _attention_head_statistics(attn: torch.Tensor, tokens):
    seq_len = attn.size(-1)
    positions = torch.arange(seq_len, dtype=torch.float32)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    stats = []
    eos_positions = [i for i, tok in enumerate(tokens) if tok == '<eos>']
    eos_idx = eos_positions[0] if eos_positions else seq_len - 1

    for head_idx, head_attn in enumerate(attn):
        diag_mean = head_attn.diagonal().mean().item()
        next_token_mean = head_attn.diagonal(offset=1).mean().item() if seq_len > 1 else 0.0
        previous_token_mean = head_attn.diagonal(offset=-1).mean().item() if seq_len > 1 else 0.0
        eos_mean = head_attn[:, eos_idx].mean().item()
        avg_distance = (head_attn * distances).sum(dim=-1).mean().item()
        entropy = (-(head_attn.clamp_min(1e-12) * head_attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
        stats.append({
            'head': head_idx,
            'diagonal_mean': diag_mean,
            'next_token_mean': next_token_mean,
            'previous_token_mean': previous_token_mean,
            'eos_mean': eos_mean,
            'avg_attention_distance': avg_distance,
            'entropy': entropy,
        })
    return stats


def _attention_heatmap_image(head_attn: torch.Tensor, tokens, head_idx: int):
    import matplotlib.pyplot as plt

    fig_size = max(6, min(14, 0.55 * len(tokens)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(head_attn.numpy(), cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_title(f'Last encoder self-attention head {head_idx}')
    ax.set_xlabel('Key token attended to')
    ax.set_ylabel('Query token')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def log_last_encoder_attention_maps(model: Transformer, val_loader: DataLoader, src_vocab, args, device: str) -> None:
    dataset = val_loader.dataset
    sample_idx = min(max(args.attention_sample_index, 0), len(dataset) - 1)
    src_tensor, tgt_tensor = dataset[sample_idx]
    src = src_tensor.unsqueeze(0).to(device)
    src_mask = make_src_mask(src, PAD_IDX)

    model.eval()
    with torch.no_grad():
        model.encode(src, src_mask)

    last_attn = model.encoder.layers[-1].self_attn.last_attn
    if last_attn is None:
        raise RuntimeError('No encoder attention weights were captured. Run model.encode before logging attention maps.')

    last_attn = last_attn[0].detach().cpu()
    tokens = _tokens_from_tensor(src_tensor, src_vocab)
    if len(tokens) != last_attn.size(-1):
        tokens = tokens[:last_attn.size(-1)]

    prefix = 'attention/last_encoder'
    wandb.log({
        f'{prefix}/sample_index': sample_idx,
        f'{prefix}/source_tokens': ' '.join(tokens),
        f'{prefix}/num_heads': last_attn.size(0),
    })

    for head_idx, head_attn in enumerate(last_attn):
        pair_table = wandb.Table(columns=['query_position', 'query_token', 'key_position', 'key_token', 'attention_weight'])
        for query_pos, query_token in enumerate(tokens):
            for key_pos, key_token in enumerate(tokens):
                pair_table.add_data(query_pos, query_token, key_pos, key_token, float(head_attn[query_pos, key_pos].item()))
        wandb.log({
            f'{prefix}/head_{head_idx}_heatmap': _attention_heatmap_image(head_attn, tokens, head_idx),
            f'{prefix}/head_{head_idx}_weights': pair_table,
        })

    stats_table = wandb.Table(columns=['head', 'diagonal_mean', 'next_token_mean', 'previous_token_mean', 'eos_mean', 'avg_attention_distance', 'entropy'])
    stats_log = {}
    for row in _attention_head_statistics(last_attn, tokens):
        stats_table.add_data(
            row['head'],
            row['diagonal_mean'],
            row['next_token_mean'],
            row['previous_token_mean'],
            row['eos_mean'],
            row['avg_attention_distance'],
            row['entropy'],
        )
        stats_log[f'{prefix}/head_{row["head"]}_diagonal_mean'] = row['diagonal_mean']
        stats_log[f'{prefix}/head_{row["head"]}_next_token_mean'] = row['next_token_mean']
        stats_log[f'{prefix}/head_{row["head"]}_avg_attention_distance'] = row['avg_attention_distance']
        stats_log[f'{prefix}/head_{row["head"]}_entropy'] = row['entropy']

    wandb.log({f'{prefix}/head_statistics': stats_table, **stats_log})


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    is_train: bool = True,
    device: str = 'cpu',
    global_step: int = 0,
    grad_log_steps: int = 0,
):
    model.train(is_train)
    total_loss, total_correct, total_tokens, total_confidence, steps = 0.0, 0, 0, 0.0, 0

    for src, tgt in data_iter:
        loss, correct, total, confidence_sum = _step_batch(model, src, tgt, loss_fn, device)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            global_step += 1
            if grad_log_steps > 0 and global_step <= grad_log_steps:
                wandb.log({
                    'train/global_step': global_step,
                    'train/batch_loss': loss.item(),
                    **get_attention_projection_grad_norms(model),
                }, step=global_step)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total
        total_confidence += confidence_sum
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    token_acc = total_correct / max(total_tokens, 1)
    prediction_confidence = total_confidence / max(total_tokens, 1)
    return avg_loss, token_acc, prediction_confidence, global_step


def greedy_decode(model: Transformer, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int, end_symbol: int, device: str = 'cpu') -> torch.Tensor:
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, PAD_IDX)
        out = model.decode(memory, src_mask.to(device), ys, tgt_mask)
        next_word = torch.argmax(out[:, -1, :], dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == end_symbol:
            break
    return ys




def _extract_tokens(token_ids: torch.Tensor, vocab):
    tokens = []
    for idx in token_ids.detach().cpu().tolist():
        tok = vocab.lookup_token(idx)
        if tok in {"<sos>", "<eos>", "<pad>"}:
            continue
        tokens.append(tok)
    return tokens


def _count_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def compute_validation_bleu(model: Transformer, val_loader: DataLoader, tgt_vocab, device: str = 'cpu', max_decode_len: int = 128) -> float:
    model.eval()
    clipped = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    cand_len = 0
    ref_len = 0

    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src_batch = src_batch.to(device)
            for i in range(src_batch.size(0)):
                src = src_batch[i:i+1]
                ref_tokens = _extract_tokens(tgt_batch[i], tgt_vocab)
                pred_ids = greedy_decode(model, src, make_src_mask(src, PAD_IDX), max_len=max_decode_len, start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device).squeeze(0)
                pred_tokens = _extract_tokens(pred_ids, tgt_vocab)

                cand_len += len(pred_tokens)
                ref_len += len(ref_tokens)

                for n in range(1, 5):
                    pred_counts = _count_ngrams(pred_tokens, n)
                    ref_counts = _count_ngrams(ref_tokens, n)
                    total[n-1] += sum(pred_counts.values())
                    clipped[n-1] += sum(min(c, ref_counts.get(g, 0)) for g, c in pred_counts.items())

    precisions = [(clipped[i] / total[i]) if total[i] > 0 else 0.0 for i in range(4)]
    if min(precisions) == 0:
        return 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1 - (ref_len / max(cand_len, 1)))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return 100.0 * bleu


def run_task_2_1(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_len,
        min_freq=args.min_freq,
        num_workers=args.num_workers,
    )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        N=args.layers,
        num_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        scale_attention=args.attention_scaling == 'scaled',
        positional_encoding_type=args.positional_encoding,
        max_len=args.max_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps) if args.scheduler == 'noam' else None
    loss_fn = LabelSmoothingLoss(vocab_size=len(tgt_vocab), pad_idx=PAD_IDX, smoothing=args.label_smoothing)

    run = wandb.init(
        project=args.project,
        entity=args.entity if args.entity else None,
        name=args.run_name,
        tags=["assignment3", args.task, f"scheduler:{args.scheduler}", f"attention_scaling:{args.attention_scaling}", f"positional_encoding:{args.positional_encoding}", f"label_smoothing:{args.label_smoothing}"],
        config=vars(args),
    )

    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_conf, global_step = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            is_train=True,
            device=device,
            global_step=global_step,
            grad_log_steps=args.grad_log_steps,
        )
        val_loss, val_acc, val_conf, _ = run_epoch(val_loader, model, loss_fn, None, None, is_train=False, device=device, global_step=global_step)
        val_bleu = compute_validation_bleu(model, val_loader, tgt_vocab, device=device, max_decode_len=args.max_len)

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch,
            'train/global_step': global_step,
            'train/loss': train_loss,
            'train/token_accuracy': train_acc,
            'train/prediction_confidence': train_conf,
            'val/loss': val_loss,
            'val/token_accuracy': val_acc,
            'val/prediction_confidence': val_conf,
            'val/bleu': val_bleu,
            'optim/lr': current_lr,
        }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.ckpt_dir, f'best_{args.scheduler}.pt'))

    if args.log_attention_maps:
        log_last_encoder_attention_maps(model, val_loader, src_vocab, args, device)

    run.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=str, default='da6401-a3')
    p.add_argument('--entity', type=str, default='')
    p.add_argument('--task', type=str, default='task2.1')
    p.add_argument('--run_name', type=str, required=True)
    p.add_argument('--scheduler', choices=['noam', 'fixed'], required=True)
    p.add_argument('--attention_scaling', choices=['scaled', 'unscaled'], default='scaled')
    p.add_argument('--positional_encoding', choices=['sinusoidal', 'learned'], default='sinusoidal')
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--grad_log_steps', type=int, default=0)
    p.add_argument('--log_attention_maps', action='store_true')
    p.add_argument('--attention_sample_index', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--min_freq', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--warmup_steps', type=int, default=4000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--ckpt_dir', type=str, default='checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_task_2_1(args)
