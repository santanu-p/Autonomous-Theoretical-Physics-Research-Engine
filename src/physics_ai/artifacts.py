from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataframe(df: pd.DataFrame, preferred_path: Path) -> Path:
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_path.suffix != ".parquet":
        preferred_path = preferred_path.with_suffix(".parquet")
    try:
        df.to_parquet(preferred_path, index=False)
        return preferred_path
    except Exception:
        fallback = preferred_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback
