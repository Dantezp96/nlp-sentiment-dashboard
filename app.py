"""NLP Sentiment Analysis Dashboard - Gradio Application."""

import os
import time
import collections
import threading
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


# ── Rate Limiter ─────────────────────────────────────────────────────

RATE_LIMITS = {
    "single":    {"calls": 15, "window": 60},   # 15 análisis/min por IP
    "batch":     {"calls": 5,  "window": 60},   # 5 lotes/min por IP
    "dashboard": {"calls": 3,  "window": 60},   # 3 dashboards/min por IP
}

_rate_data: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
_rate_lock = threading.Lock()


def check_rate_limit(ip: str, action: str) -> tuple[bool, str]:
    """Returns (allowed, error_message). Thread-safe sliding window."""
    limit = RATE_LIMITS[action]
    max_calls = limit["calls"]
    window = limit["window"]
    now = time.time()

    with _rate_lock:
        timestamps = _rate_data[ip][action]
        # Purge expired entries
        _rate_data[ip][action] = [t for t in timestamps if now - t < window]
        if len(_rate_data[ip][action]) >= max_calls:
            oldest = _rate_data[ip][action][0]
            wait = int(window - (now - oldest)) + 1
            return False, f"⚠️ Límite alcanzado. Intenta de nuevo en {wait}s."
        _rate_data[ip][action].append(now)
        return True, ""


def get_ip(request: gr.Request) -> str:
    """Extract client IP, respecting X-Forwarded-For from proxies."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Tab 1: Analyze Text ─────────────────────────────────────────────

def analyze_single(text: str, request: gr.Request):
    """Analiza un solo texto."""
    ip = get_ip(request)
    allowed, err = check_rate_limit(ip, "single")
    if not allowed:
        return {}, err

    if not text or not text.strip():
        return {}, "Ingresa algún texto para analizar."
    text = preprocess_text(text)
    scores = predict_single(text)
    top_label, top_conf = get_top_sentiment(scores)
    label_map = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo"}
    summary = f"**{label_map.get(top_label, top_label.capitalize())}** ({top_conf:.1%} de confianza)"
    return scores, summary


def analyze_batch(text_input: str, file_input, request: gr.Request):
    """Analiza múltiples textos desde textarea o archivo CSV."""
    ip = get_ip(request)
    allowed, err = check_rate_limit(ip, "batch")
    if not allowed:
        return None, None, None

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

    rows = []
    label_map = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo"}
    for text, scores in zip(texts, results):
        top_label, top_conf = get_top_sentiment(scores)
        rows.append({
            "Texto": text[:100] + ("..." if len(text) > 100 else ""),
            "Sentimiento": label_map.get(top_label, top_label.capitalize()),
            "Confianza": f"{top_conf:.1%}",
            "Positivo": f"{scores.get('positive', 0):.3f}",
            "Neutral": f"{scores.get('neutral', 0):.3f}",
            "Negativo": f"{scores.get('negative', 0):.3f}",
        })

    result_df = pd.DataFrame(rows)
    dist_chart = create_sentiment_distribution(results)
    csv_path = "/tmp/sentiment_results.csv"
    result_df.to_csv(csv_path, index=False)

    return result_df, dist_chart, csv_path


# ── Tab 2: Dashboard ────────────────────────────────────────────────

def run_dashboard(dataset_name: str, sample_size: int, request: gr.Request):
    """Genera dashboard completo sobre un dataset de muestra."""
    ip = get_ip(request)
    allowed, err = check_rate_limit(ip, "dashboard")
    if not allowed:
        return (err,) + (None,) * 11

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
        "Texto": [t[:80] + "..." if len(t) > 80 else t for t in texts[:20]],
        "Sentimiento": [s.capitalize() for s in sentiments[:20]],
        "Confianza": [f"{max(r.values()):.1%}" for r in results[:20]],
    })

    return (
        f"### {stats['total']} textos analizados",
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
    title="Dashboard de Análisis de Sentimientos NLP",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="emerald",
        neutral_hue="slate",
    ),
    css=custom_css,
) as demo:

    gr.Markdown(
        "# 🔍 Dashboard de Análisis de Sentimientos NLP\n"
        "Análisis de sentimientos en tiempo real impulsado por **RoBERTa** "
        "(cardiffnlp/twitter-roberta-base-sentiment-latest)"
    )

    # ── Tab 1: Analizar ──
    with gr.Tab("Analizar Texto"):
        gr.Markdown("### Análisis de Texto Individual")
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Ingresa un texto",
                    placeholder="Escribe una oración para analizar su sentimiento...",
                    lines=3,
                )
            with gr.Column(scale=1):
                analyze_btn = gr.Button("Analizar", variant="primary", size="lg")

        with gr.Row():
            label_output = gr.Label(label="Puntuaciones de Sentimiento", num_top_classes=3)
            summary_output = gr.Markdown()

        analyze_btn.click(analyze_single, inputs=text_input, outputs=[label_output, summary_output])
        text_input.submit(analyze_single, inputs=text_input, outputs=[label_output, summary_output])

        gr.Markdown("---\n### Análisis por Lotes")
        gr.Markdown("Pega múltiples líneas o sube un archivo CSV con columna `text`.")

        with gr.Row():
            batch_text = gr.Textbox(
                label="Pega textos (uno por línea)",
                lines=5,
                placeholder="Primer texto a analizar\nSegundo texto\nTercer texto...",
            )
            batch_file = gr.File(label="O sube un CSV", file_types=[".csv", ".txt"])

        batch_btn = gr.Button("Analizar Lote", variant="primary")

        batch_df = gr.Dataframe(label="Resultados", interactive=False)
        batch_chart = gr.Plot(label="Distribución")
        batch_csv = gr.File(label="Descargar Resultados")

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
                ["¡Me encantó la atención al cliente, super rápidos y amables!"],
                ["El producto llegó en mal estado y el soporte no respondió nunca."],
            ],
            inputs=text_input,
        )

    # ── Tab 2: Dashboard ──
    with gr.Tab("Dashboard"):
        gr.Markdown("### Análisis de Dataset Precargado")
        gr.Markdown("Selecciona un dataset y tamaño de muestra para generar un dashboard completo.")

        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=get_available_datasets(),
                value=get_available_datasets()[0] if get_available_datasets() else None,
                label="Dataset",
            )
            sample_slider = gr.Slider(
                minimum=50, maximum=2000, value=500, step=50,
                label="Tamaño de Muestra",
            )
            dash_btn = gr.Button("Generar Dashboard", variant="primary")

        with gr.Row():
            stat_total = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_pos = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_neg = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_neu = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_conf = gr.Markdown("### -", elem_classes=["stat-card"])

        with gr.Row():
            gr.Markdown("**Total de Textos**", elem_classes=["stat-card"])
            gr.Markdown("**% Positivos**", elem_classes=["stat-card"])
            gr.Markdown("**% Negativos**", elem_classes=["stat-card"])
            gr.Markdown("**% Neutrales**", elem_classes=["stat-card"])
            gr.Markdown("**Confianza Promedio**", elem_classes=["stat-card"])

        with gr.Row():
            dash_dist = gr.Plot(label="Distribución de Sentimientos")
            dash_conf = gr.Plot(label="Puntuaciones de Confianza")

        with gr.Row():
            dash_wc_pos = gr.Image(label="Nube de Palabras Positivas", type="pil")
            dash_wc_neg = gr.Image(label="Nube de Palabras Negativas", type="pil")

        dash_top_words = gr.Plot(label="Palabras Más Frecuentes por Sentimiento")
        dash_temporal = gr.Plot(label="Tendencias de Sentimiento en el Tiempo")
        dash_sample = gr.Dataframe(label="Predicciones de Muestra", interactive=False)

        dash_btn.click(
            run_dashboard,
            inputs=[dataset_dd, sample_slider],
            outputs=[
                stat_total, stat_pos, stat_neg, stat_neu, stat_conf,
                dash_dist, dash_conf, dash_wc_pos, dash_wc_neg,
                dash_top_words, dash_temporal, dash_sample,
            ],
        )

    # ── Tab 3: Acerca de ──
    with gr.Tab("Acerca de"):
        gr.Markdown("""
## Acerca de Este Proyecto

Este **Dashboard de Análisis de Sentimientos NLP** es una herramienta de clasificación de sentimientos
en tiempo real, construida por **Omar Daniel Zorro** como parte de su portafolio de Data Science & ML.

### Modelo
- **Arquitectura**: RoBERTa-base ajustado con ~124M tweets
- **ID del Modelo**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Clases**: Negativo, Neutral, Positivo
- **Inferencia**: ~100ms por texto en CPU

### Stack Tecnológico
| Componente | Tecnología |
|-----------|-----------|
| Framework UI | Gradio |
| Modelo ML | HuggingFace Transformers |
| Deep Learning | PyTorch (CPU) |
| Visualización | Matplotlib, WordCloud |
| Procesamiento | Pandas, NumPy |
| Despliegue | Railway (Docker) |

### Límites de uso (por IP)
| Acción | Límite |
|--------|--------|
| Análisis individual | 15 por minuto |
| Análisis por lotes | 5 por minuto |
| Generación de dashboard | 3 por minuto |

### Datasets
- **Tweets**: TweetEval Sentiment dataset (Barbieri et al., 2020) — 2,000 tweets de muestra
- **Reseñas**: Amazon Polarity dataset — 1,000 reseñas de productos

### Enlaces
- [Portafolio](https://frontend-mauve-seven-17.vercel.app)
- [Repositorio GitHub](https://github.com/Dantezp96/nlp-sentiment-dashboard)

---
*Construido con HuggingFace Transformers y Gradio*
        """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
