import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from tokenizer import SMILESTokenizer, PAD_IDX

MAX_LEN = 120 # obtained from sequence length distribution from data


class ZINCDataset(Dataset):
    # Load zinc-druglike-cano.csv on demand using byte offsets to avoid loading the entire file into memory

    def __init__(self, csv_path, tokenizer, max_len=MAX_LEN, max_rows=None):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_rows = max_rows  # cap rows 
        self.offsets = None
        self._index()

    def _index(self):
        # One-pass scan: collect offsets into a plain list, then compact into numpy once
        offsets = []
        with open(self.csv_path, "rb") as f:
            f.readline()  # skip header
            pbar = tqdm(desc="Indexing", unit=" rows", total=self.max_rows)
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
                    pbar.update(1)
                    if self.max_rows and len(offsets) >= self.max_rows:
                        break
            pbar.close()
        self.offsets = np.array(offsets, dtype=np.int64)
        del offsets  

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.csv_path, "rb") as f:
            f.seek(int(self.offsets[idx]))  
            smi = f.readline().decode("utf-8").strip()
        indices = self.tokenizer.encode(smi, add_sos=True, add_eos=True, max_len=self.max_len)
        seq = torch.tensor(indices, dtype=torch.long)
        return {"input": seq[:-1], "target": seq[1:], "length": len(seq) - 1}


def collate_fn(batch):
    inputs  = [item["input"]  for item in batch]
    targets = [item["target"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    return {
        "input":   pad_sequence(inputs,  batch_first=True, padding_value=PAD_IDX),
        "target":  pad_sequence(targets, batch_first=True, padding_value=PAD_IDX),
        "lengths": lengths,
    }


def build_dataloaders(csv_path, tokenizer, batch_size=512, val_split=0.1, test_split=0.1, num_workers=0, seed=42, max_rows=None):
    # Build dataset and split into train/val/test using a single random permutation of indices (no copying of data)
    dataset = ZINCDataset(csv_path, tokenizer, max_rows=max_rows)
    n = len(dataset)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)  # keep as tensor

    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    test_idx  = perm[:n_test]
    val_idx   = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    del perm 

    pin = torch.cuda.is_available()

    def make_loader(idx, shuffle):
        return DataLoader(
            Subset(dataset, idx), 
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin,
        )

    return make_loader(train_idx, True), make_loader(val_idx, False), make_loader(test_idx, False)