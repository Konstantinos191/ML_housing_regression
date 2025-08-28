from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import pandas as pd

@dataclass
class ColumnSpec:
    target: str
    numeric: Sequence[str]
    categorical: Sequence[str]

def split_xy(df: pd.DataFrame, spec: ColumnSpec) -> Tuple[pd.DataFrame, pd.Series]:
    needed = list(spec.numeric) + list(spec.categorical) + [spec.target]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
    X = df[list(spec.numeric) + list(spec.categorical)].copy()
    y = df[spec.target].copy()
    return X, y
