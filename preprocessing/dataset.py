import torch
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
import os

WINDOW_SIZE = 10
FEATURE_COLUMNS = ['row_hit_rate', 'consec_hits', 'dist_from_last_hit', 'diff_accesses']
IDENTIFIERS = ['channel', 'rank', 'bankgroup', 'bank']

class DRAMBankWindowPreprocessor:
    def __init__(self, filepaths, window_size=10, append_from=None):
        self.window_size = window_size
        self.addr_fields = ['channel', 'rank', 'bankgroup', 'bank', 'row', 'column']
        self.stats_fields = ['row_hit_rate', 'consec_hits', 'dist_from_last_hit', 'diff_accesses']
        self.identifiers = ['channel', 'rank', 'bankgroup', 'bank']

        if append_from and os.path.exists(append_from):
            print(f"Loading existing samples from {append_from}")
            self.samples = torch.load(append_from)
        else:
            self.samples = []

        for filepath in filepaths:
            df = pd.read_csv(filepath)

            for col in self.addr_fields + self.stats_fields:
                df[col] = df[col].astype('float32')

            for _, group in df.groupby(self.identifiers):
                group = group.reset_index(drop=True)

                for i in range(len(group) - window_size):
                    addr_window = group.iloc[i:i + window_size][self.addr_fields].values
                    stats_window = group.iloc[i:i + window_size][self.stats_fields].values
                    label_row = group.iloc[i + window_size]
                    label = {'closed': 0, 'open': 1}[label_row['label']]

                    self.samples.append((
                        torch.tensor(addr_window, dtype=torch.float32),
                        torch.tensor(stats_window, dtype=torch.float32),
                        torch.tensor(label, dtype=torch.long)
                    ))

    def save(self, path="data_new/all.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.samples, path)
        print(f"Saved preprocessed dataset to {path}")

class CachedDRAMDataset(Dataset):
    def __init__(self, path):
        self.samples = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]