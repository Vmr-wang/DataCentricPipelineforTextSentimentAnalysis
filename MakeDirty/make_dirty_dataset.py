#!/usr/bin/env python3
# make_dirty_reviews_2023.py
#
# Input:  Amazon Reviews 2023 raw review JSONL (one JSON per line)
# Output: corrupted ("dirty") JSONL
#
# Dependencies:
#   pip install pandas numpy jenga==0.0.1a1
# Note: PyPI jenga==0.0.1a1 requires Python < 3.10.

import argparse
import json
import math
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from jenga.basis import DataCorruption
from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import GaussianNoise, Scaling


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunked(it: Iterable[Dict], chunk_size: int) -> Iterable[List[Dict]]:
    buf: List[Dict] = []
    for rec in it:
        buf.append(rec)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def normalize_review_record(rec: Dict) -> Dict:
    # HF dataset schema uses helpful_vote (int). Keep it as the canonical name.
    out = dict(rec)

    # If some upstream code has helpful_votes, map it back.
    if "helpful_vote" not in out and "helpful_votes" in out:
        out["helpful_vote"] = out["helpful_votes"]

    return out


def to_jsonable(v):
    """
    Convert pandas/numpy scalars and containers to JSON-serializable Python types.
    Also map NaN/NA to None.
    """
    if v is pd.NA:
        return None

    # numpy scalar -> python scalar
    if isinstance(v, np.generic):
        v = v.item()

    # NaN -> None
    if isinstance(v, float) and math.isnan(v):
        return None

    # Containers
    if isinstance(v, list):
        return [to_jsonable(x) for x in v]
    if isinstance(v, tuple):
        return [to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: to_jsonable(val) for k, val in v.items()}

    return v


class DuplicateRows(DataCorruption):
    """Add duplicated rows (common in raw logs / merges)."""

    def __init__(self, fraction: float, seed: Optional[int] = None):
        super().__init__()
        self.fraction = float(fraction)
        self.seed = seed

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(deep=True)
        n = int(round(len(df) * self.fraction))
        if n <= 0:
            return df
        dup = df.sample(n=n, replace=True, random_state=self.seed)
        return pd.concat([df, dup], ignore_index=True)


class TextGarbling(DataCorruption):
    """
    Insert dirty characters in text fields:
    - add leading/trailing spaces
    - replace a few chars with '�' or accented chars
    - sometimes insert control/BOM chars
    """

    def __init__(self, column: str, fraction: float, seed: int = 0):
        super().__init__()
        self.column = column
        self.fraction = float(fraction)
        self.rng = random.Random(seed)

    def _garble(self, s: str) -> str:
        if not s:
            return s

        if self.rng.random() < 0.5:
            s = (" " * self.rng.randint(1, 3)) + s
        if self.rng.random() < 0.5:
            s = s + (" " * self.rng.randint(1, 3))

        chars = list(s)
        k = max(1, int(round(0.01 * len(chars))))
        for _ in range(k):
            i = self.rng.randrange(len(chars))
            chars[i] = "�" if self.rng.random() < 0.7 else self.rng.choice(["á", "é", "ö", "ß"])
        s2 = "".join(chars)

        if self.rng.random() < 0.3:
            i = self.rng.randrange(len(s2) + 1)
            s2 = s2[:i] + self.rng.choice(["\u0000", "\u0008", "\ufeff"]) + s2[i:]
        return s2

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(deep=True)
        if self.column not in df.columns:
            return df

        # Ensure string writes do not conflict with pandas dtype
        df[self.column] = df[self.column].astype("object")

        idx = df.index[df[self.column].notna()].tolist()
        m = int(round(len(idx) * self.fraction))
        if m <= 0:
            return df

        chosen = self.rng.sample(idx, k=min(m, len(idx)))
        for i in chosen:
            v = df.at[i, self.column]
            if isinstance(v, str):
                df.at[i, self.column] = self._garble(v)
        return df


class TypeErrors(DataCorruption):
    """
    Convert numeric/bool fields into strings like 'N/A', '5', '5.0', 'true', 'false'.
    """

    def __init__(self, column: str, fraction: float, seed: int = 0):
        super().__init__()
        self.column = column
        self.fraction = float(fraction)
        self.rng = random.Random(seed)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(deep=True)
        if self.column not in df.columns:
            return df

        # Ensure string writes do not conflict with pandas dtype
        df[self.column] = df[self.column].astype("object")

        idx = df.index[df[self.column].notna()].tolist()
        m = int(round(len(idx) * self.fraction))
        if m <= 0:
            return df

        chosen = self.rng.sample(idx, k=min(m, len(idx)))
        for i in chosen:
            v = df.at[i, self.column]
            if self.rng.random() < 0.2:
                df.at[i, self.column] = "N/A"
                continue

            if isinstance(v, (bool, np.bool_)):
                df.at[i, self.column] = "true" if bool(v) else "false"
            elif isinstance(v, (int, float, np.integer, np.floating)):
                if self.rng.random() < 0.5:
                    df.at[i, self.column] = str(int(round(float(v))))
                else:
                    df.at[i, self.column] = f"{float(v):.1f}"
            else:
                df.at[i, self.column] = str(v)

        return df


def build_corruptions(seed: int, strength: float) -> List[DataCorruption]:
    # strength is a multiplier for fractions; keep it >= 0
    s = max(0.0, float(strength))

    return [
        # Missingness
        MissingValues("text", fraction=0.03 * s, na_value=None, missingness="MCAR"),
        MissingValues("title", fraction=0.01 * s, na_value=None, missingness="MCAR"),
        MissingValues("rating", fraction=0.002 * s, na_value=None, missingness="MCAR"),

        # Numeric issues (helpful_vote)
        Scaling("helpful_vote", fraction=0.002 * s, sampling="CAR"),
        GaussianNoise("helpful_vote", fraction=0.01 * s, sampling="CAR"),

        # Key mix-up (asin vs parent_asin)
        SwappedValues("asin", fraction=0.001 * s, sampling="CAR", swap_with="parent_asin"),

        # Text garbling
        TextGarbling("title", fraction=0.01 * s, seed=seed),
        TextGarbling("text", fraction=0.005 * s, seed=seed + 1),

        # Type errors
        TypeErrors("rating", fraction=0.002 * s, seed=seed + 2),
        TypeErrors("helpful_vote", fraction=0.002 * s, seed=seed + 3),
        TypeErrors("verified_purchase", fraction=0.001 * s, seed=seed + 4),

        # Duplicates
        DuplicateRows(fraction=0.001 * s, seed=seed),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw review JSONL (e.g., raw/review_categories/Gift_Cards.jsonl)")
    ap.add_argument("--output", required=True, help="output dirty JSONL")
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=1.0, help="multiplier for all corruption fractions")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cols = [
        "rating", "title", "text", "images",
        "asin", "parent_asin", "user_id", "timestamp",
        "verified_purchase", "helpful_vote",
    ]

    corruptions = build_corruptions(args.seed, args.strength)

    total_in = 0
    total_out = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for chunk in chunked(iter_jsonl(args.input), args.chunk_size):
            norm = [normalize_review_record(r) for r in chunk]

            df = pd.DataFrame([{c: r.get(c) for c in cols} for r in norm])

            # Normalize numeric types so numerical corruptions can run
            # Force float to avoid int dtype conflicts when adding noise
            if "rating" in df.columns:
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")
            if "helpful_vote" in df.columns:
                df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").astype("float64")

            df2 = df
            for c in corruptions:
                df2 = c.transform(df2)

            base_n = len(norm)
            new_n = len(df2)

            # Update existing rows
            for i in range(min(base_n, new_n)):
                r = dict(norm[i])
                for c in cols:
                    v = df2.iloc[i][c] if c in df2.columns else r.get(c)
                    r[c] = to_jsonable(v)
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Write extra rows created by corruptions (duplicates)
            for i in range(base_n, new_n):
                r = dict(norm[base_n - 1]) if base_n > 0 else {}
                for c in cols:
                    v = df2.iloc[i][c] if c in df2.columns else r.get(c)
                    r[c] = to_jsonable(v)
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

            total_in += base_n
            total_out += new_n

    print(f"done. input_rows={total_in}, output_rows={total_out}")


if __name__ == "__main__":
    main()
