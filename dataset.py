from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import spacy


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


@dataclass
class Vocab:
    stoi: dict
    itos: List[str]
    unk_idx: int = UNK_IDX
    pad_idx: int = PAD_IDX
    sos_idx: int = SOS_IDX
    eos_idx: int = EOS_IDX

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = 'train',
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        max_length: int = 128,
        min_freq: int = 2,
    ):
        self.split = split
        self.max_length = max_length
        self.min_freq = min_freq

        self.raw_data = load_dataset('bentrevett/multi30k', split=split)

        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        self.samples = self.process_data()

    def tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self) -> Tuple[Vocab, Vocab]:
        train_data = load_dataset('bentrevett/multi30k', split='train')

        src_counter, tgt_counter = Counter(), Counter()
        for ex in train_data:
            src_counter.update(self.tokenize_de(ex['de']))
            tgt_counter.update(self.tokenize_en(ex['en']))

        src_itos = list(SPECIAL_TOKENS)
        tgt_itos = list(SPECIAL_TOKENS)

        src_itos.extend([tok for tok, c in src_counter.items() if c >= self.min_freq and tok not in SPECIAL_TOKENS])
        tgt_itos.extend([tok for tok, c in tgt_counter.items() if c >= self.min_freq and tok not in SPECIAL_TOKENS])

        src_stoi = {tok: i for i, tok in enumerate(src_itos)}
        tgt_stoi = {tok: i for i, tok in enumerate(tgt_itos)}

        return Vocab(src_stoi, src_itos), Vocab(tgt_stoi, tgt_itos)

    def numericalize(self, tokens: List[str], vocab: Vocab) -> List[int]:
        ids = [vocab.lookup_index(tok) for tok in tokens]
        ids = ids[: self.max_length - 2]
        return [vocab.sos_idx] + ids + [vocab.eos_idx]

    def process_data(self):
        processed = []
        for ex in self.raw_data:
            src_toks = self.tokenize_de(ex['de'])
            tgt_toks = self.tokenize_en(ex['en'])
            src_ids = torch.tensor(self.numericalize(src_toks, self.src_vocab), dtype=torch.long)
            tgt_ids = torch.tensor(self.numericalize(tgt_toks, self.tgt_vocab), dtype=torch.long)
            processed.append((src_ids, tgt_ids))
        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_idx: int = PAD_IDX):
    src_batch = [x[0] for x in batch]
    tgt_batch = [x[1] for x in batch]
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_pad, tgt_pad


def build_dataloaders(batch_size: int = 64, max_length: int = 128, min_freq: int = 2, num_workers: int = 0):
    train_ds = Multi30kDataset(split='train', max_length=max_length, min_freq=min_freq)
    val_ds = Multi30kDataset(split='validation', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab, max_length=max_length, min_freq=min_freq)
    test_ds = Multi30kDataset(split='test', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab, max_length=max_length, min_freq=min_freq)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, PAD_IDX))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, PAD_IDX))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, PAD_IDX))

    return train_loader, val_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab
