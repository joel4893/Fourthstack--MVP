import os
from typing import Optional, Callable

import torch
import pandas as pd
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """A simple Dataset wrapper around a JSONL file produced by the
    synthetic generator.

    It loads the JSONL into a DataFrame and exposes rows as float32 tensors.
    Optional `transform` can be provided to apply on-the-fly to each tensor.
    """

    def __init__(self, jsonl_path: str, transform: Optional[Callable] = None):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL path not found: {jsonl_path}")

        df = pd.read_json(jsonl_path, lines=True)
        # Ensure numeric values where possible; non-numeric columns will be coerced to NaN
        try:
            values = df.values.astype(float)
        except Exception:
            # fall back to attempting numeric conversion per-column
            values = pd.to_numeric(df.values, errors="coerce")

        self.data = torch.tensor(values, dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.data))

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.data[idx]
        if self.transform:
            return self.transform(row)
        return row


__all__ = ["SyntheticDataset"]
