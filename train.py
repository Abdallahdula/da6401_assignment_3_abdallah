import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


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
            pad_mask = target.eq(self.pad_idx)
            true_dist[pad_mask] = 0
        loss = -(true_dist * log_probs).sum(dim=1)
        non_pad = ~target.eq(self.pad_idx)
        return loss[non_pad].mean()


def run_epoch(data_iter, model: Transformer, loss_fn: nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler=None, epoch_num: int = 0, is_train: bool = True, device: str = 'cpu') -> float:
    model.train(is_train)
    total = 0.0
    steps = 0
    for src, tgt in data_iter:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask, tgt_mask = make_src_mask(src), make_tgt_mask(tgt_in)
        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total += loss.item(); steps += 1
    return total / max(steps, 1)


def greedy_decode(model: Transformer, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int, end_symbol: int, device: str = 'cpu') -> torch.Tensor:
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)
        out = model.decode(memory, src_mask.to(device), ys, tgt_mask)
        next_word = torch.argmax(out[:, -1, :], dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == end_symbol:
            break
    return ys


def evaluate_bleu(model: Transformer, test_dataloader: DataLoader, tgt_vocab, device: str = 'cpu', max_len: int = 100) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu
    except Exception:
        return 0.0
    model.eval()
    refs, hyps = [], []
    sos = getattr(tgt_vocab, 'sos_idx', 2)
    eos = getattr(tgt_vocab, 'eos_idx', 3)
    pad = getattr(tgt_vocab, 'pad_idx', 1)
    def itos(i):
        if hasattr(tgt_vocab, 'itos'):
            return tgt_vocab.itos[i]
        return tgt_vocab.lookup_token(i)
    with torch.no_grad():
        for src, tgt in test_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            for i in range(src.size(0)):
                s = src[i:i+1]
                sm = make_src_mask(s, pad)
                pred = greedy_decode(model, s, sm, max_len, sos, eos, device=device)[0].tolist()
                pred_toks = [itos(x) for x in pred if x not in (sos, eos, pad)]
                gold = tgt[i].tolist()
                gold_toks = [itos(x) for x in gold if x not in (sos, eos, pad)]
                hyps.append(pred_toks)
                refs.append([gold_toks])
    return corpus_bleu(refs, hyps) * 100


def save_checkpoint(model: Transformer, optimizer: torch.optim.Optimizer, scheduler, epoch: int, path: str = 'checkpoint.pt') -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'd_model': model.d_model,
            'N': model.N,
            'num_heads': model.num_heads,
            'd_ff': model.d_ff,
            'dropout': model.dropout,
        }
    }, path)


def load_checkpoint(path: str, model: Transformer, optimizer: Optional[torch.optim.Optimizer] = None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return int(ckpt['epoch'])


def run_training_experiment() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    run_training_experiment()
