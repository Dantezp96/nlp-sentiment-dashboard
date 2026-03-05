"""Utilities for text preprocessing and CSV handling."""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def preprocess_text(text: str) -> str:
    """Light text cleaning."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # remove # but keep word
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_csv(file_path: str) -> pd.DataFrame:
    """Read CSV and detect the text column automatically."""
    df = pd.read_csv(file_path)
    if "text" in df.columns:
        return df
    # Heuristic: pick the column with longest average string length
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) == 0:
        raise ValueError("No text columns found in CSV")
    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in str_cols}
    text_col = max(avg_lengths, key=avg_lengths.get)
    df = df.rename(columns={text_col: "text"})
    return df


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a pre-prepared sample dataset."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
    return pd.read_csv(path)


def get_available_datasets() -> list[str]:
    """List available sample datasets."""
    return [f.stem for f in DATA_DIR.glob("*.csv")]
