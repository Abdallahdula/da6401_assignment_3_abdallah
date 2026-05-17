import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import build_dataloaders, PAD_IDX


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
    return loss, correct, total


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
    total_loss, total_correct, total_tokens, steps = 0.0, 0, 0, 0

    for src, tgt in data_iter:
        loss, correct, total = _step_batch(model, src, tgt, loss_fn, device)
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
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    token_acc = total_correct / max(total_tokens, 1)
    return avg_loss, token_acc, global_step


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
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps) if args.scheduler == 'noam' else None
    loss_fn = LabelSmoothingLoss(vocab_size=len(tgt_vocab), pad_idx=PAD_IDX, smoothing=0.1)

    run = wandb.init(
        project=args.project,
        entity=args.entity if args.entity else None,
        name=args.run_name,
        tags=["assignment3", args.task, f"scheduler:{args.scheduler}", f"attention_scaling:{args.attention_scaling}"],
        config=vars(args),
    )

    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, global_step = run_epoch(
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
        val_loss, val_acc, _ = run_epoch(val_loader, model, loss_fn, None, None, is_train=False, device=device, global_step=global_step)

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch,
            'train/global_step': global_step,
            'train/loss': train_loss,
            'train/token_accuracy': train_acc,
            'val/loss': val_loss,
            'val/token_accuracy': val_acc,
            'optim/lr': current_lr,
        }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.ckpt_dir, f'best_{args.scheduler}.pt'))

    run.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=str, default='da6401-a3')
    p.add_argument('--entity', type=str, default='')
    p.add_argument('--task', type=str, default='task2.1')
    p.add_argument('--run_name', type=str, required=True)
    p.add_argument('--scheduler', choices=['noam', 'fixed'], required=True)
    p.add_argument('--attention_scaling', choices=['scaled', 'unscaled'], default='scaled')
    p.add_argument('--grad_log_steps', type=int, default=0)
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
