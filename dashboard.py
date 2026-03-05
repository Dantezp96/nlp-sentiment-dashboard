"""Dashboard analytics and visualization functions."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image

COLORS = {"negative": "#ef4444", "neutral": "#f59e0b", "positive": "#22c55e"}


def create_sentiment_distribution(results: list[dict]) -> plt.Figure:
    """Pie chart of sentiment distribution."""
    sentiments = [max(r, key=r.get) for r in results]
    counts = Counter(sentiments)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [COLORS.get(l, "#6366f1") for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=[f"{l.capitalize()}\n({v})" for l, v in zip(labels, sizes)],
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11, "color": "white"}
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")

    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    return fig


def create_confidence_histogram(results: list[dict]) -> plt.Figure:
    """Histogram of max confidence scores."""
    confidences = [max(r.values()) for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=20, color="#8b5cf6", edgecolor="#1a1a2e", alpha=0.85)
    ax.set_xlabel("Confidence Score", fontsize=11, color="white")
    ax.set_ylabel("Count", fontsize=11, color="white")
    ax.set_title("Confidence Distribution", fontsize=14, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    return fig


def create_wordcloud(texts: list[str], sentiments: list[str], target: str = "positive") -> Image.Image:
    """Generate word cloud for a specific sentiment."""
    filtered = [t for t, s in zip(texts, sentiments) if s == target]
    if not filtered:
        img = Image.new("RGB", (600, 300), color=(26, 26, 46))
        return img

    text = " ".join(filtered)
    color = COLORS.get(target, "#6366f1")

    wc = WordCloud(
        width=600, height=300,
        background_color="#1a1a2e",
        colormap="Purples" if target == "neutral" else ("Greens" if target == "positive" else "Reds"),
        max_words=80,
        min_font_size=10,
        max_font_size=60,
    ).generate(text)

    return wc.to_image()


def create_top_words_chart(texts: list[str], sentiments: list[str]) -> plt.Figure:
    """Bar chart of top words per sentiment."""
    import re
    stopwords = {"the", "a", "an", "is", "it", "to", "and", "of", "in", "for", "on",
                 "that", "this", "with", "was", "are", "be", "have", "has", "had",
                 "not", "but", "at", "from", "or", "by", "as", "do", "if", "my", "i",
                 "me", "so", "we", "you", "he", "she", "they", "them", "its", "no",
                 "just", "very", "can", "will", "about", "up", "out", "all", "been",
                 "would", "there", "their", "what", "when", "who", "how", "than", "more"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#1a1a2e")

    for idx, sentiment in enumerate(["negative", "neutral", "positive"]):
        ax = axes[idx]
        filtered = [t for t, s in zip(texts, sentiments) if s == sentiment]
        words = []
        for t in filtered:
            words.extend([w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", t) if w.lower() not in stopwords])

        if words:
            top = Counter(words).most_common(10)
            labels, values = zip(*reversed(top))
            color = COLORS.get(sentiment, "#6366f1")
            ax.barh(labels, values, color=color, alpha=0.85)
        ax.set_title(sentiment.capitalize(), fontsize=12, fontweight="bold", color="white")
        ax.tick_params(colors="white", labelsize=9)
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle("Top Words by Sentiment", fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    return fig


def create_temporal_trends(results: list[dict], timestamps: list) -> plt.Figure:
    """Line chart of sentiment over time."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "sentiment": [max(r, key=r.get) for r in results],
        "confidence": [max(r.values()) for r in results],
    })
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for sentiment in ["positive", "neutral", "negative"]:
        if sentiment in daily.columns:
            ax.plot(daily.index, daily[sentiment], color=COLORS[sentiment],
                    label=sentiment.capitalize(), linewidth=2, alpha=0.85)

    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax.set_title("Sentiment Trends Over Time", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Date", fontsize=11, color="white")
    ax.set_ylabel("Count", fontsize=11, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def compute_summary_stats(results: list[dict]) -> dict:
    """Compute summary statistics from prediction results."""
    sentiments = [max(r, key=r.get) for r in results]
    confidences = [max(r.values()) for r in results]
    counts = Counter(sentiments)
    total = len(results)

    return {
        "total": total,
        "positive_pct": round(counts.get("positive", 0) / total * 100, 1),
        "negative_pct": round(counts.get("negative", 0) / total * 100, 1),
        "neutral_pct": round(counts.get("neutral", 0) / total * 100, 1),
        "avg_confidence": round(np.mean(confidences) * 100, 1),
    }
