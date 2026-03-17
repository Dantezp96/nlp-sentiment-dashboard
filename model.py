"""Sentiment analysis model using HuggingFace Transformers."""

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully!")

LABELS = [config.id2label[i] for i in range(config.num_labels)]


def predict_single(text: str) -> dict[str, float]:
    """Analyze sentiment of a single text.

    Returns dict like {'negative': 0.05, 'neutral': 0.15, 'positive': 0.80}
    """
    if not text or not text.strip():
        return {label: 0.0 for label in LABELS}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
    return {config.id2label[i]: float(scores[i]) for i in range(len(scores))}


def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
    """Analyze sentiment of multiple texts in batches."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [t for t in texts[i:i + batch_size] if t and t.strip()]
        if not batch:
            results.extend([{label: 0.0 for label in LABELS}] * len(texts[i:i + batch_size]))
            continue
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
        for row in scores:
            results.append({config.id2label[j]: float(row[j]) for j in range(len(row))})
    return results


def get_top_sentiment(scores: dict[str, float]) -> tuple[str, float]:
    """Get the dominant sentiment and its confidence."""
    top_label = max(scores, key=scores.get)
    return top_label, scores[top_label]
