"""NLP Sentiment Analysis Dashboard - Gradio Application."""

import os
import gradio as gr
import pandas as pd
from model import predict_single, predict_batch, get_top_sentiment, LABELS
from dashboard import (
    create_sentiment_distribution,
    create_confidence_histogram,
    create_wordcloud,
    create_top_words_chart,
    create_temporal_trends,
    compute_summary_stats,
)
from utils import preprocess_text, parse_csv, load_sample_dataset, get_available_datasets


# ── Tab 1: Analyze Text ─────────────────────────────────────────────

def analyze_single(text: str):
    """Analyze a single text input."""
    if not text or not text.strip():
        return {}, "Enter some text to analyze"
    text = preprocess_text(text)
    scores = predict_single(text)
    top_label, top_conf = get_top_sentiment(scores)
    summary = f"**{top_label.capitalize()}** ({top_conf:.1%} confidence)"
    return scores, summary


def analyze_batch(text_input: str, file_input):
    """Analyze multiple texts from textarea or CSV file."""
    texts = []

    if file_input is not None:
        df = parse_csv(file_input)
        texts = df["text"].dropna().astype(str).tolist()
    elif text_input and text_input.strip():
        texts = [line.strip() for line in text_input.strip().split("\n") if line.strip()]

    if not texts:
        return None, None, None

    texts = [preprocess_text(t) for t in texts]
    results = predict_batch(texts)

    # Build results dataframe
    rows = []
    for text, scores in zip(texts, results):
        top_label, top_conf = get_top_sentiment(scores)
        rows.append({
            "Text": text[:100] + ("..." if len(text) > 100 else ""),
            "Sentiment": top_label.capitalize(),
            "Confidence": f"{top_conf:.1%}",
            "Positive": f"{scores.get('positive', 0):.3f}",
            "Neutral": f"{scores.get('neutral', 0):.3f}",
            "Negative": f"{scores.get('negative', 0):.3f}",
        })

    result_df = pd.DataFrame(rows)
    dist_chart = create_sentiment_distribution(results)
    csv_path = "/tmp/sentiment_results.csv"
    result_df.to_csv(csv_path, index=False)

    return result_df, dist_chart, csv_path


# ── Tab 2: Dashboard ────────────────────────────────────────────────

def run_dashboard(dataset_name: str, sample_size: int):
    """Run full dashboard analysis on a sample dataset."""
    df = load_sample_dataset(dataset_name)
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    texts = [preprocess_text(t) for t in df["text"].tolist()]
    results = predict_batch(texts)
    sentiments = [max(r, key=r.get) for r in results]
    stats = compute_summary_stats(results)
    timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else None

    dist_fig = create_sentiment_distribution(results)
    conf_fig = create_confidence_histogram(results)
    wc_pos = create_wordcloud(texts, sentiments, "positive")
    wc_neg = create_wordcloud(texts, sentiments, "negative")
    top_words_fig = create_top_words_chart(texts, sentiments)
    temporal_fig = create_temporal_trends(results, timestamps) if timestamps else None

    sample_df = pd.DataFrame({
        "Text": [t[:80] + "..." if len(t) > 80 else t for t in texts[:20]],
        "Sentiment": [s.capitalize() for s in sentiments[:20]],
        "Confidence": [f"{max(r.values()):.1%}" for r in results[:20]],
    })

    return (
        f"### {stats['total']} texts analyzed",
        f"### {stats['positive_pct']}%",
        f"### {stats['negative_pct']}%",
        f"### {stats['neutral_pct']}%",
        f"### {stats['avg_confidence']}%",
        dist_fig,
        conf_fig,
        wc_pos,
        wc_neg,
        top_words_fig,
        temporal_fig,
        sample_df,
    )


# ── CSS ──────────────────────────────────────────────────────────────

custom_css = """
.gradio-container { max-width: 1200px !important; }
.stat-card { text-align: center; padding: 10px; }
"""

# ── Build UI ─────────────────────────────────────────────────────────

with gr.Blocks(
    title="NLP Sentiment Dashboard",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="emerald",
        neutral_hue="slate",
    ),
    css=custom_css,
) as demo:

    gr.Markdown(
        "# 🔍 NLP Sentiment Analysis Dashboard\n"
        "Real-time sentiment analysis powered by **RoBERTa** (cardiffnlp/twitter-roberta-base-sentiment-latest)"
    )

    # ── Tab 1: Analyze ──
    with gr.Tab("Analyze Text"):
        gr.Markdown("### Single Text Analysis")
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Enter text",
                    placeholder="Type a sentence to analyze its sentiment...",
                    lines=3,
                )
            with gr.Column(scale=1):
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Row():
            label_output = gr.Label(label="Sentiment Scores", num_top_classes=3)
            summary_output = gr.Markdown()

        analyze_btn.click(analyze_single, inputs=text_input, outputs=[label_output, summary_output])
        text_input.submit(analyze_single, inputs=text_input, outputs=[label_output, summary_output])

        gr.Markdown("---\n### Batch Analysis")
        gr.Markdown("Paste multiple lines or upload a CSV file with a `text` column.")

        with gr.Row():
            batch_text = gr.Textbox(
                label="Paste texts (one per line)",
                lines=5,
                placeholder="First text to analyze\nSecond text\nThird text...",
            )
            batch_file = gr.File(label="Or upload CSV", file_types=[".csv", ".txt"])

        batch_btn = gr.Button("Analyze Batch", variant="primary")

        batch_df = gr.Dataframe(label="Results", interactive=False)
        batch_chart = gr.Plot(label="Distribution")
        batch_csv = gr.File(label="Download Results")

        batch_btn.click(
            analyze_batch,
            inputs=[batch_text, batch_file],
            outputs=[batch_df, batch_chart, batch_csv],
        )

        gr.Examples(
            examples=[
                ["I absolutely love this product! Best purchase I've ever made."],
                ["The service was okay, nothing special but not terrible either."],
                ["Terrible experience. The item arrived broken and customer support was unhelpful."],
                ["Just finished watching the new movie. The cinematography was stunning but the plot was predictable."],
            ],
            inputs=text_input,
        )

    # ── Tab 2: Dashboard ──
    with gr.Tab("Dashboard"):
        gr.Markdown("### Pre-loaded Dataset Analysis")
        gr.Markdown("Select a dataset and sample size to generate a full sentiment analysis dashboard.")

        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=get_available_datasets(),
                value=get_available_datasets()[0] if get_available_datasets() else None,
                label="Dataset",
            )
            sample_slider = gr.Slider(
                minimum=50, maximum=2000, value=500, step=50,
                label="Sample Size",
            )
            dash_btn = gr.Button("Generate Dashboard", variant="primary")

        # Stats row
        with gr.Row():
            stat_total = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_pos = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_neg = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_neu = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_conf = gr.Markdown("### -", elem_classes=["stat-card"])

        with gr.Row():
            gr.Markdown("**Total Texts**", elem_classes=["stat-card"])
            gr.Markdown("**Positive %**", elem_classes=["stat-card"])
            gr.Markdown("**Negative %**", elem_classes=["stat-card"])
            gr.Markdown("**Neutral %**", elem_classes=["stat-card"])
            gr.Markdown("**Avg Confidence**", elem_classes=["stat-card"])

        with gr.Row():
            dash_dist = gr.Plot(label="Sentiment Distribution")
            dash_conf = gr.Plot(label="Confidence Scores")

        with gr.Row():
            dash_wc_pos = gr.Image(label="Positive Word Cloud", type="pil")
            dash_wc_neg = gr.Image(label="Negative Word Cloud", type="pil")

        dash_top_words = gr.Plot(label="Top Words by Sentiment")
        dash_temporal = gr.Plot(label="Sentiment Trends Over Time")
        dash_sample = gr.Dataframe(label="Sample Predictions", interactive=False)

        dash_btn.click(
            run_dashboard,
            inputs=[dataset_dd, sample_slider],
            outputs=[
                stat_total, stat_pos, stat_neg, stat_neu, stat_conf,
                dash_dist, dash_conf, dash_wc_pos, dash_wc_neg,
                dash_top_words, dash_temporal, dash_sample,
            ],
        )

    # ── Tab 3: About ──
    with gr.Tab("About"):
        gr.Markdown("""
## About This Project

This **NLP Sentiment Analysis Dashboard** is a real-time sentiment classification tool built
by **Omar Daniel Zorro** as part of his Data Science & ML portfolio.

### Model
- **Architecture**: RoBERTa-base fine-tuned on ~124M tweets
- **Model ID**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Classes**: Negative, Neutral, Positive
- **Inference**: ~100ms per text on CPU

### Tech Stack
| Component | Technology |
|-----------|-----------|
| UI Framework | Gradio |
| ML Model | HuggingFace Transformers |
| Deep Learning | PyTorch (CPU) |
| Visualization | Matplotlib, WordCloud |
| Data Processing | Pandas, NumPy |
| Deployment | Railway (Docker) |

### Datasets
- **Tweets**: TweetEval Sentiment dataset (Barbieri et al., 2020) — 2,000 sample tweets
- **Reviews**: Amazon Polarity dataset — 1,000 sample product reviews

### Links
- [Portfolio](https://frontend-mauve-seven-17.vercel.app)
- [GitHub Repository](https://github.com/Dantezp96/nlp-sentiment-dashboard)

---
*Built with HuggingFace Transformers and Gradio*
        """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
