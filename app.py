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
    "single":    {"calls": 15, "window": 60},
    "batch":     {"calls": 5,  "window": 60},
    "dashboard": {"calls": 3,  "window": 60},
}

_rate_data: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
_rate_lock = threading.Lock()


def check_rate_limit(ip: str, action: str) -> tuple[bool, str]:
    limit = RATE_LIMITS[action]
    now = time.time()
    with _rate_lock:
        _rate_data[ip][action] = [t for t in _rate_data[ip][action] if now - t < limit["window"]]
        if len(_rate_data[ip][action]) >= limit["calls"]:
            wait = int(limit["window"] - (now - _rate_data[ip][action][0])) + 1
            return False, f"⏳ Usaste demasiado el analizador. Espera **{wait} segundos** e intenta de nuevo."
        _rate_data[ip][action].append(now)
        return True, ""


def get_ip(request: gr.Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Helpers ───────────────────────────────────────────────────────────

EMOJI_MAP = {"positive": "😊", "neutral": "😐", "negative": "😠"}
LABEL_MAP  = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo"}

def confidence_phrase(conf: float) -> str:
    """Convierte la confianza numérica en lenguaje natural."""
    if conf >= 0.90:
        return "muy seguro 🎯"
    if conf >= 0.75:
        return "bastante seguro ✅"
    if conf >= 0.55:
        return "más o menos seguro 🤔"
    return "poco seguro, podría equivocarse ⚠️"


def build_result_card(top_label: str, top_conf: float, scores: dict) -> str:
    """Genera un resultado explicado en lenguaje simple."""
    emoji = EMOJI_MAP[top_label]
    label = LABEL_MAP[top_label]
    phrase = confidence_phrase(top_conf)

    # Barra visual de porcentaje (10 bloques)
    filled = round(top_conf * 10)
    bar = "█" * filled + "░" * (10 - filled)

    # Descripción del sentimiento detectado
    descriptions = {
        "positive": "Las palabras del texto transmiten **alegría, satisfacción o aprobación**.",
        "neutral":  "El texto es **informativo o equilibrado** — no expresa emociones fuertes.",
        "negative": "Las palabras del texto transmiten **disgusto, queja o desaprobación**.",
    }

    other_scores = "\n".join(
        f"- {EMOJI_MAP[k]} **{LABEL_MAP[k]}**: {v:.0%}"
        for k, v in sorted(scores.items(), key=lambda x: -x[1])
        if k != top_label
    )

    return f"""## {emoji} Sentimiento detectado: **{label}**

`{bar}` **{top_conf:.0%}** de confianza — el modelo está {phrase}

{descriptions[top_label]}

<details>
<summary>Ver todos los puntajes</summary>

- {emoji} **{label}**: {top_conf:.0%}
{other_scores}

> 💡 Los tres puntajes siempre suman 100%. El modelo asigna probabilidad a cada categoría y elige la más alta.
</details>
"""


# ── Tab 1: Analizar ───────────────────────────────────────────────────

def analyze_single(text: str, request: gr.Request):
    ip = get_ip(request)
    allowed, err = check_rate_limit(ip, "single")
    if not allowed:
        return {}, err

    if not text or not text.strip():
        return {}, "✏️ Escribe algo arriba y presiona **Analizar**."

    clean = preprocess_text(text)
    scores = predict_single(clean)
    top_label, top_conf = get_top_sentiment(scores)
    card = build_result_card(top_label, top_conf, scores)
    return scores, card


def analyze_batch(text_input: str, file_input, request: gr.Request):
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
    for text, scores in zip(texts, results):
        top_label, top_conf = get_top_sentiment(scores)
        rows.append({
            "Texto": text[:100] + ("..." if len(text) > 100 else ""),
            "Sentimiento": f"{EMOJI_MAP[top_label]} {LABEL_MAP[top_label]}",
            "Confianza": f"{top_conf:.0%}",
            "😊 Positivo": f"{scores.get('positive', 0):.0%}",
            "😐 Neutral":  f"{scores.get('neutral', 0):.0%}",
            "😠 Negativo": f"{scores.get('negative', 0):.0%}",
        })

    result_df = pd.DataFrame(rows)
    dist_chart = create_sentiment_distribution(results)
    csv_path = "/tmp/sentiment_results.csv"
    result_df.to_csv(csv_path, index=False)
    return result_df, dist_chart, csv_path


# ── Tab 2: Dashboard ──────────────────────────────────────────────────

def run_dashboard(dataset_name: str, sample_size: int, request: gr.Request):
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

    dist_fig      = create_sentiment_distribution(results)
    conf_fig      = create_confidence_histogram(results)
    wc_pos        = create_wordcloud(texts, sentiments, "positive")
    wc_neg        = create_wordcloud(texts, sentiments, "negative")
    top_words_fig = create_top_words_chart(texts, sentiments)
    temporal_fig  = create_temporal_trends(results, timestamps) if timestamps else None

    sample_df = pd.DataFrame({
        "Texto": [t[:80] + "..." if len(t) > 80 else t for t in texts[:20]],
        "Sentimiento": [f"{EMOJI_MAP[s]} {LABEL_MAP[s]}" for s in sentiments[:20]],
        "Confianza": [f"{max(r.values()):.0%}" for r in results[:20]],
    })

    return (
        f"### 📝 {stats['total']} textos analizados",
        f"### 😊 {stats['positive_pct']}%",
        f"### 😠 {stats['negative_pct']}%",
        f"### 😐 {stats['neutral_pct']}%",
        f"### 🎯 {stats['avg_confidence']}%",
        dist_fig, conf_fig, wc_pos, wc_neg, top_words_fig, temporal_fig, sample_df,
    )


# ── CSS ───────────────────────────────────────────────────────────────

custom_css = """
.gradio-container { max-width: 1100px !important; }
.stat-card        { text-align: center; padding: 8px; }
.tip-box          { background: #f0f4ff; border-left: 4px solid #7c3aed;
                    padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
"""

# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Analizador de Sentimientos NLP",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="emerald", neutral_hue="slate"),
    css=custom_css,
) as demo:

    gr.Markdown("""
# 🧠 Analizador de Sentimientos con Inteligencia Artificial

> **¿Qué hace esta app?**
> Lee cualquier texto y detecta si expresa una emoción **positiva** 😊, **negativa** 😠 o **neutral** 😐.
> Es como enseñarle a una computadora a "leer entre líneas".

**Modelo usado:** RoBERTa — entrenado con 124 millones de tweets reales de todo el mundo.
""")

    # ── Tab 1 ──
    with gr.Tab("✏️ Analizar un texto"):

        gr.Markdown("""
### ¿Cómo se usa?
1. **Escribe** cualquier oración en el cuadro de abajo (en español o inglés)
2. Presiona **Analizar** o la tecla **Enter**
3. La IA te dirá qué emoción detectó y qué tan segura está
""")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📝 Tu texto aquí",
                    placeholder='Ej: "Me encantó la película, fue increíble!" o "El servicio fue pésimo."',
                    lines=3,
                )
            with gr.Column(scale=1):
                analyze_btn = gr.Button("🔍 Analizar", variant="primary", size="lg")

        with gr.Row():
            label_output = gr.Label(
                label="📊 Probabilidad por categoría (los 3 puntajes suman 100%)",
                num_top_classes=3,
            )
            summary_output = gr.Markdown(value="*El resultado aparecerá aquí...*")

        analyze_btn.click(analyze_single, inputs=text_input, outputs=[label_output, summary_output])
        text_input.submit(analyze_single, inputs=text_input, outputs=[label_output, summary_output])

        gr.Markdown("""
---
### 💡 Prueba con estos ejemplos
Haz clic en cualquiera para cargarlo automáticamente:
""")
        gr.Examples(
            label="Ejemplos listos para probar",
            examples=[
                ["¡Me encantó la atención, fueron súper amables y rápidos! 🌟"],
                ["El producto llegó roto y el soporte nunca respondió. Pésimo."],
                ["El paquete llegó el martes por la mañana."],
                ["I absolutely love this! Best purchase I've ever made."],
                ["Terrible experience. The item arrived broken."],
                ["The weather today is cloudy with a chance of rain."],
            ],
            inputs=text_input,
        )

        gr.Markdown("""
---
### 📦 Analizar varios textos a la vez
¿Tienes una lista de comentarios, reseñas o tweets? Pégalos aquí (uno por línea) o sube un archivo CSV con columna `text`.
""")

        with gr.Row():
            batch_text = gr.Textbox(
                label="Pega varios textos (uno por línea)",
                lines=5,
                placeholder="Primer texto\nSegundo texto\nTercer texto...",
            )
            batch_file = gr.File(label="O sube un CSV", file_types=[".csv", ".txt"])

        batch_btn = gr.Button("📊 Analizar todos", variant="primary")

        batch_df    = gr.Dataframe(label="Resultados", interactive=False)
        batch_chart = gr.Plot(label="¿Cuántos positivos, negativos y neutrales hay?")
        batch_csv   = gr.File(label="⬇️ Descargar resultados en CSV")

        batch_btn.click(
            analyze_batch,
            inputs=[batch_text, batch_file],
            outputs=[batch_df, batch_chart, batch_csv],
        )

    # ── Tab 2 ──
    with gr.Tab("📊 Dashboard de dataset"):

        gr.Markdown("""
### Analiza cientos de textos de un golpe

Aquí puedes cargar un dataset completo (conjunto de datos real) y ver estadísticas generales:
cuántos textos son positivos, cuáles palabras aparecen más, cómo cambia el sentimiento en el tiempo, etc.

> **Tamaño de muestra:** cuántos textos del dataset quieres analizar. Más textos = más tarda, pero resultados más completos.
""")

        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=get_available_datasets(),
                value=get_available_datasets()[0] if get_available_datasets() else None,
                label="📂 Elige un dataset",
            )
            sample_slider = gr.Slider(
                minimum=50, maximum=2000, value=500, step=50,
                label="Cantidad de textos a analizar",
            )
            dash_btn = gr.Button("🚀 Generar Dashboard", variant="primary")

        gr.Markdown("#### Resumen")
        with gr.Row():
            stat_total = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_pos   = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_neg   = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_neu   = gr.Markdown("### -", elem_classes=["stat-card"])
            stat_conf  = gr.Markdown("### -", elem_classes=["stat-card"])

        with gr.Row():
            gr.Markdown("**📝 Total**",           elem_classes=["stat-card"])
            gr.Markdown("**😊 Positivos**",       elem_classes=["stat-card"])
            gr.Markdown("**😠 Negativos**",       elem_classes=["stat-card"])
            gr.Markdown("**😐 Neutrales**",       elem_classes=["stat-card"])
            gr.Markdown("**🎯 Confianza prom.**", elem_classes=["stat-card"])

        gr.Markdown("""
#### 📈 Distribución y confianza
> **Distribución:** cuántos textos caen en cada categoría.
> **Confianza:** qué tan seguro estuvo el modelo en sus predicciones (0% = adivinando, 100% = certeza total).
""")
        with gr.Row():
            dash_dist = gr.Plot(label="¿Cuántos textos son positivos, negativos y neutrales?")
            dash_conf = gr.Plot(label="¿Qué tan seguro estuvo el modelo en cada predicción?")

        gr.Markdown("""
#### ☁️ Nubes de palabras
> Las palabras más grandes son las que aparecen más seguido en cada tipo de texto.
> Sirven para entender **qué temas generan emociones positivas o negativas**.
""")
        with gr.Row():
            dash_wc_pos = gr.Image(label="😊 Palabras más usadas en textos POSITIVOS", type="pil")
            dash_wc_neg = gr.Image(label="😠 Palabras más usadas en textos NEGATIVOS", type="pil")

        gr.Markdown("""
#### 🏆 Palabras clave y tendencias en el tiempo
> ¿Qué palabras "delatan" el sentimiento de un texto?
> Si ves "love" o "amazing" en positivos y "broken" o "terrible" en negativos, el modelo aprendió bien.
""")
        dash_top_words = gr.Plot(label="Palabras que más distinguen cada sentimiento")
        dash_temporal  = gr.Plot(label="¿Cómo cambia el sentimiento a lo largo del tiempo?")
        dash_sample    = gr.Dataframe(label="Muestra de predicciones individuales", interactive=False)

        dash_btn.click(
            run_dashboard,
            inputs=[dataset_dd, sample_slider],
            outputs=[
                stat_total, stat_pos, stat_neg, stat_neu, stat_conf,
                dash_dist, dash_conf, dash_wc_pos, dash_wc_neg,
                dash_top_words, dash_temporal, dash_sample,
            ],
        )

    # ── Tab 3 ──
    with gr.Tab("🎓 ¿Cómo funciona?"):
        gr.Markdown("""
## ¿Cómo sabe la IA qué emoción tiene un texto?

### 🧩 Paso 1 — El texto se convierte en números
Las computadoras no entienden palabras, solo números.
Cada palabra se transforma en un vector (lista de números) que captura su significado.

> *Ejemplo: "feliz" y "contento" tendrán vectores muy parecidos porque significan algo similar.*

---

### 🔬 Paso 2 — El modelo analiza el contexto
El modelo usado aquí se llama **RoBERTa** — una red neuronal entrenada con **124 millones de tweets**.
Aprendió a reconocer patrones: qué combinaciones de palabras suelen ser positivas, negativas o neutras.

> *Es como si hubiera leído millones de comentarios de personas reales y aprendido sus patrones.*

---

### 🎯 Paso 3 — Devuelve 3 probabilidades
Para cada texto el modelo calcula:
- 😊 Probabilidad de que sea **positivo**
- 😐 Probabilidad de que sea **neutral**
- 😠 Probabilidad de que sea **negativo**

Los tres valores **siempre suman 100%**. El sentimiento ganador es el de mayor probabilidad.

---

### 📏 ¿Qué significa la "confianza"?
La confianza es qué tan seguro está el modelo. Imagina que preguntaras a 100 personas:

| Confianza | Significado |
|-----------|-------------|
| 95% | 95 de 100 personas dirían lo mismo |
| 70% | 70 de 100 — hay algo de duda |
| 51% | Casi un empate — el modelo "adivina" |

---

### 🤔 ¿El modelo siempre acierta?
**No.** Los modelos de IA cometen errores, especialmente con:
- Sarcasmo ("Qué *maravilloso* servicio..." 🙄)
- Negaciones dobles ("no está mal" = positivo)
- Contexto cultural o jerga local

Por eso siempre es bueno revisar los resultados con ojo crítico.

---

### 🛠️ Tecnologías usadas
| Componente | Para qué sirve |
|-----------|----------------|
| **RoBERTa** (HuggingFace) | Modelo de IA que clasifica el sentimiento |
| **Gradio** | La interfaz visual que estás usando ahora |
| **PyTorch** | Motor que ejecuta el modelo de IA |
| **Matplotlib** | Genera las gráficas |
| **WordCloud** | Genera las nubes de palabras |
| **Railway** | Servidor donde corre todo esto |

---

*Construido por **Omar Daniel Zorro** · [Portafolio](https://frontend-mauve-seven-17.vercel.app) · [GitHub](https://github.com/Dantezp96/nlp-sentiment-dashboard)*
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
