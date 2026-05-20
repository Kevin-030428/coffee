from __future__ import annotations

import pandas as pd
import streamlit as st

from src.coffee_catalog import COFFEE_MENU
from src.coffee_pipelines import (
    extract_coffee_entities,
    load_ner_pipeline,
    load_sentiment_pipeline,
    predict_sentiment,
    summarize_entities,
)


st.set_page_config(
    page_title="Starbucks Review Intelligence System",
    page_icon="S",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --starbucks-green: #006241;
        --deep-green: #1e3932;
        --mint: #d4e9e2;
        --cream: #f7f2e8;
        --coffee: #6f4e37;
        --ink: #1f2933;
        --muted: #65737e;
        --line: rgba(30, 57, 50, 0.14);
    }

    .stApp {
        background:
            linear-gradient(180deg, rgba(247, 242, 232, 0.92), rgba(255, 255, 255, 0.98) 42%),
            radial-gradient(circle at top left, rgba(0, 98, 65, 0.12), transparent 34%);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3932 0%, #0e2a23 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f6fff9 !important;
    }

    [data-testid="stSidebar"] .stCaption {
        color: rgba(246, 255, 249, 0.72) !important;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .hero {
        border: 1px solid var(--line);
        background:
            linear-gradient(135deg, rgba(0, 98, 65, 0.96), rgba(30, 57, 50, 0.94)),
            linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
        border-radius: 6px;
        padding: 28px 30px 26px;
        box-shadow: 0 18px 42px rgba(30, 57, 50, 0.15);
        margin-bottom: 22px;
    }

    .hero-kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #d4e9e2;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.35rem;
        line-height: 1.1;
        margin: 0 0 10px;
        letter-spacing: 0;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.84);
        max-width: 760px;
        font-size: 1.03rem;
        margin: 0;
    }

    .pipeline-strip {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 0 0 20px;
    }

    .pipeline-card,
    .result-card,
    .note-card {
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.86);
        border-radius: 6px;
        padding: 16px 18px;
        box-shadow: 0 10px 28px rgba(30, 57, 50, 0.08);
    }

    .pipeline-label {
        color: var(--starbucks-green);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    .pipeline-title {
        font-weight: 750;
        color: var(--deep-green);
        font-size: 1.05rem;
        margin-bottom: 5px;
    }

    .pipeline-copy {
        color: var(--muted);
        font-size: 0.9rem;
        margin: 0;
    }

    .result-card.positive {
        border-left: 5px solid #00754a;
    }

    .result-card.negative {
        border-left: 5px solid #a23b2a;
    }

    .result-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.72rem;
        font-weight: 800;
    }

    .result-value {
        color: var(--deep-green);
        font-size: 1.8rem;
        font-weight: 800;
        margin: 2px 0 2px;
    }

    .result-sub {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .coffee-chip {
        display: inline-block;
        border: 1px solid rgba(0, 98, 65, 0.22);
        background: #eef7f2;
        color: var(--deep-green);
        padding: 5px 9px;
        border-radius: 999px;
        margin: 3px 5px 3px 0;
        font-size: 0.86rem;
        font-weight: 650;
    }

    .section-title {
        font-size: 1.12rem;
        color: var(--deep-green);
        font-weight: 800;
        margin: 8px 0 10px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 4px;
        border: 1px solid var(--starbucks-green);
        background: var(--starbucks-green);
        color: #ffffff;
        font-weight: 750;
        min-height: 44px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--deep-green);
        background: var(--deep-green);
        color: #ffffff;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.88);
        border-radius: 6px;
        padding: 12px 14px;
    }

    div[data-testid="stTabs"] button {
        color: var(--deep-green);
        font-weight: 700;
    }

    textarea {
        border-radius: 6px !important;
    }

    @media (max-width: 760px) {
        .pipeline-strip {
            grid-template-columns: 1fr;
        }
        .hero {
            padding: 22px 20px;
        }
        .hero h1 {
            font-size: 1.78rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading sentiment model...")
def cached_sentiment_pipeline():
    return load_sentiment_pipeline()


@st.cache_resource(show_spinner="Loading coffee NER model...")
def cached_ner_pipeline():
    return load_ner_pipeline()


def analyze_review(review: str, sentiment_pipe, ner_pipe) -> dict[str, object]:
    sentiment = predict_sentiment(review, sentiment_pipe)
    entities = extract_coffee_entities(review, ner_pipe)
    return {
        "review": review,
        "sentiment": sentiment.label,
        "sentiment_confidence": round(sentiment.score, 4),
        "sentiment_runtime_ms": round(sentiment.runtime_ms, 2),
        "coffee_items": summarize_entities(entities),
        "entity_count": len(entities),
        "sentiment_source": sentiment.source,
        "entity_source": entities[0].source if entities else "no entity detected",
    }


def render_pipeline_overview() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Starbucks customer intelligence</div>
            <h1>Starbucks Review Intelligence System</h1>
            <p>Classify Starbucks review sentiment and extract mentioned coffee products with two fine-tuned Hugging Face pipelines trained in Google Colab.</p>
        </div>
        <div class="pipeline-strip">
            <div class="pipeline-card">
                <div class="pipeline-label">Pipeline 1</div>
                <div class="pipeline-title">Sentiment classification</div>
                <p class="pipeline-copy">Hugging Face text-classification model predicts Positive or Negative review sentiment.</p>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-label">Pipeline 2</div>
                <div class="pipeline-title">Coffee named entity recognition</div>
                <p class="pipeline-copy">Hugging Face token-classification model extracts Starbucks coffee items mentioned in reviews.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_coffee_chips(items: str) -> str:
    if not items or items == "No coffee item detected":
        return '<span class="coffee-chip">No coffee item detected</span>'
    return "".join(f'<span class="coffee-chip">{item.strip()}</span>' for item in items.split(","))


def render_single_result(result: dict[str, object]) -> None:
    sentiment = str(result["sentiment"])
    tone = "positive" if sentiment == "Positive" else "negative"
    confidence = float(result["sentiment_confidence"])
    runtime = float(result["sentiment_runtime_ms"])
    coffee_items = str(result["coffee_items"])

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        st.markdown(
            f"""
            <div class="result-card {tone}">
                <div class="result-label">Sentiment</div>
                <div class="result-value">{sentiment}</div>
                <div class="result-sub">Pipeline 1 result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Confidence</div>
                <div class="result-value">{confidence:.2%}</div>
                <div class="result-sub">Model probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Runtime</div>
                <div class="result-value">{runtime:.1f} ms</div>
                <div class="result-sub">Pipeline 1 latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Detected coffee products</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="note-card">
            {render_coffee_chips(coffee_items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_review(sentiment_pipe, ner_pipe) -> None:
    default_text = (
        "The oat milk latte was creamy and the cold brew tasted smooth, "
        "but the caramel macchiato was too sweet."
    )
    st.markdown('<div class="section-title">Single review analysis</div>', unsafe_allow_html=True)
    review = st.text_area("Customer review", value=default_text, height=170)

    if st.button("Analyze review", type="primary", use_container_width=True):
        result = analyze_review(review, sentiment_pipe, ner_pipe)
        render_single_result(result)

        with st.expander("View raw prediction record"):
            st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)


def render_batch_review(sentiment_pipe, ner_pipe) -> None:
    st.markdown('<div class="section-title">Batch CSV analysis</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file with a review column", type=["csv"])
    if uploaded_file is None:
        st.markdown(
            """
            <div class="note-card">
                Upload a CSV file with a <strong>review</strong> column. The app will return sentiment,
                confidence, runtime, and detected coffee products for every row.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = pd.read_csv(uploaded_file)
    if "review" not in df.columns:
        st.error("CSV must contain a `review` column.")
        return

    st.caption(f"Loaded {len(df)} rows from the uploaded CSV.")
    if st.button("Analyze CSV", type="primary", use_container_width=True):
        rows = [analyze_review(str(text), sentiment_pipe, ner_pipe) for text in df["review"].fillna("")]
        result_df = pd.DataFrame(rows)
        positive_count = int((result_df["sentiment"] == "Positive").sum())
        negative_count = int((result_df["sentiment"] == "Negative").sum())

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows analyzed", len(result_df))
        col_b.metric("Positive reviews", positive_count)
        col_c.metric("Negative reviews", negative_count)

        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="coffee_review_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


render_pipeline_overview()

with st.sidebar:
    st.title("Project")
    st.write("Starbucks")
    st.link_button("Company website", "https://www.starbucks.com/", use_container_width=True)
    st.divider()
    st.subheader("Model stack")
    st.write("Pipeline 1: text-classification")
    st.write("Pipeline 2: token-classification")
    st.write("Training platform: Google Colab")
    st.divider()
    st.subheader("Coffee vocabulary")
    st.write(", ".join(COFFEE_MENU))
    st.divider()
    st.caption("Strict mode: the app only loads the Colab fine-tuned Hugging Face models Cry1008/coffee-sentiment and Cry1008/coffee-ner.")

sentiment_pipeline = cached_sentiment_pipeline()
ner_pipeline = cached_ner_pipeline()

tab_single, tab_batch = st.tabs(["Single review", "Batch CSV"])
with tab_single:
    render_single_review(sentiment_pipeline, ner_pipeline)
with tab_batch:
    render_batch_review(sentiment_pipeline, ner_pipeline)
