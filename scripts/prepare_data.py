"""One-time script to prepare sample datasets for the dashboard."""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_tweets():
    """Download and prepare tweet sentiment dataset."""
    print("Loading tweet_eval sentiment dataset...")
    ds = load_dataset("tweet_eval", "sentiment", split="train")
    df = pd.DataFrame(ds)

    # Map labels: 0=negative, 1=neutral, 2=positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df["label"] = df["label"].map(label_map)

    # Sample 2000 rows
    df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)

    # Add synthetic timestamps (spread over 30 days)
    base = pd.Timestamp("2025-01-01")
    df["timestamp"] = [base + pd.Timedelta(hours=int(h)) for h in np.linspace(0, 30 * 24, len(df))]

    df[["text", "label", "timestamp"]].to_csv(OUTPUT_DIR / "tweets_sample.csv", index=False)
    print(f"Saved {len(df)} tweets to tweets_sample.csv")
    print(f"Distribution:\n{df['label'].value_counts()}")


def prepare_reviews():
    """Download and prepare product review dataset."""
    print("Loading amazon_polarity dataset (sample)...")
    ds = load_dataset("amazon_polarity", split="test", streaming=True)

    rows = []
    for i, item in enumerate(ds):
        if i >= 5000:
            break
        rows.append({
            "text": item["content"][:500],
            "label": "positive" if item["label"] == 1 else "negative",
        })

    df = pd.DataFrame(rows)
    df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)

    # Add synthetic timestamps
    base = pd.Timestamp("2025-02-01")
    df["timestamp"] = [base + pd.Timedelta(hours=int(h)) for h in np.linspace(0, 30 * 24, len(df))]

    df[["text", "label", "timestamp"]].to_csv(OUTPUT_DIR / "reviews_sample.csv", index=False)
    print(f"Saved {len(df)} reviews to reviews_sample.csv")
    print(f"Distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    prepare_tweets()
    print()
    prepare_reviews()
    print("\nDone!")
