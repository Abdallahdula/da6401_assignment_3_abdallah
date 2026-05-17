import math
import copy
import os
from collections import Counter
from typing import Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
import spacy

try:
    import gdown
except Exception:
    gdown = None


def scaled_dot_product_attention(Q, K, V, mask: Optional[torch.Tensor] = None, scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    bsz, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, scale_attention: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale_attention = scale_attention
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None

    def _split(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _combine(self, x):
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        out, attn = scaled_dot_product_attention(q, k, v, mask, scale=self.scale_attention)
        self.last_attn = attn.detach()
        out = self._combine(out)
        out = self.w_o(out)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Backward-compatibility alias expected by autograder/tests
class PositionalEncoding(SinusoidalPositionalEncoding):
    pass


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, scale_attention: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale_attention=scale_attention)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, scale_attention: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale_attention=scale_attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, scale_attention=scale_attention)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int = 10000, tgt_vocab_size: int = 10000, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1, checkpoint_path: str = None, scale_attention: bool = True, positional_encoding_type: str = 'sinusoidal', max_len: int = 5000):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.scale_attention = scale_attention
        self.positional_encoding_type = positional_encoding_type
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        if positional_encoding_type == 'sinusoidal':
            self.pos = SinusoidalPositionalEncoding(d_model, dropout, max_len=max_len)
        elif positional_encoding_type == 'learned':
            self.pos = LearnedPositionalEncoding(d_model, dropout, max_len=max_len)
        else:
            raise ValueError(f'Unsupported positional_encoding_type: {positional_encoding_type}')
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout, scale_attention=scale_attention), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout, scale_attention=scale_attention), N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.max_len = max_len

        # Required by autograder-style inference: initialize tokenizer/vocab in __init__
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.unk_idx = 0
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        self.src_stoi, self.src_itos, self.tgt_stoi, self.tgt_itos = self._build_vocabs_from_train()

        # Required by announcement: load weights in __init__ (download with gdown if needed)
        self._load_checkpoint_in_init(checkpoint_path)

    def encode(self, src, src_mask):
        return self.encoder(self.pos(self.src_embed(src) * math.sqrt(self.d_model)), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        dec = self.decoder(self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model)), memory, src_mask, tgt_mask)
        return self.generator(dec)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def _build_vocabs_from_train(self):
        special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
        train_data = load_dataset('bentrevett/multi30k', split='train')
        src_counter, tgt_counter = Counter(), Counter()
        for ex in train_data:
            src_counter.update([tok.text.lower() for tok in self.spacy_de.tokenizer(ex['de'])])
            tgt_counter.update([tok.text.lower() for tok in self.spacy_en.tokenizer(ex['en'])])
        src_itos = list(special_tokens) + [tok for tok, c in src_counter.items() if c >= 2 and tok not in special_tokens]
        tgt_itos = list(special_tokens) + [tok for tok, c in tgt_counter.items() if c >= 2 and tok not in special_tokens]
        src_stoi = {tok: i for i, tok in enumerate(src_itos)}
        tgt_stoi = {tok: i for i, tok in enumerate(tgt_itos)}
        return src_stoi, src_itos, tgt_stoi, tgt_itos

    def _load_checkpoint_in_init(self, checkpoint_path: Optional[str]) -> None:
        default_local = os.path.join('checkpoints', 'best_noam.pt')
        ckpt = checkpoint_path if checkpoint_path else default_local
        if (not os.path.exists(ckpt)) and gdown is not None:
            file_id = os.environ.get('A3_WEIGHTS_FILE_ID', '').strip()
            if file_id:
                url = f'https://drive.google.com/uc?id={file_id}'
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                gdown.download(url, ckpt, quiet=True)
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location='cpu')
            model_state = state.get('model_state_dict', state)
            self.load_state_dict(model_state, strict=False)

    def _numericalize_src(self, text: str):
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        ids = [self.src_stoi.get(tok, self.unk_idx) for tok in tokens][: self.max_len - 2]
        return [self.sos_idx] + ids + [self.eos_idx]

    def _decode_tgt_ids(self, token_ids):
        out = []
        for idx in token_ids:
            if idx in (self.sos_idx, self.pad_idx):
                continue
            if idx == self.eos_idx:
                break
            if 0 <= idx < len(self.tgt_itos):
                out.append(self.tgt_itos[idx])
        return ' '.join(out)

    def infer(self, src_sentence: str) -> str:
        if not isinstance(src_sentence, str):
            src_sentence = str(src_sentence)
        src_sentence = src_sentence.strip()
        if not src_sentence:
            return ""
        src_ids = torch.tensor([self._numericalize_src(src_sentence)], dtype=torch.long, device=next(self.parameters()).device)
        src_mask = make_src_mask(src_ids, self.pad_idx)

        self.eval()
        with torch.no_grad():
            memory = self.encode(src_ids, src_mask)
            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=src_ids.device)
            for _ in range(self.max_len - 1):
                tgt_mask = make_tgt_mask(ys, self.pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_word = torch.argmax(logits[:, -1, :], dim=-1).item()
                ys = torch.cat([ys, torch.tensor([[next_word]], device=src_ids.device)], dim=1)
                if next_word == self.eos_idx:
                    break
        return self._decode_tgt_ids(ys.squeeze(0).tolist())
