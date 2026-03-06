import streamlit as st
import pandas as pd
import nltk
import os
import ssl
from nltk import tokenize
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px
import plotly.graph_objects as go
import textrazor
from sentence_transformers import SentenceTransformer, util
import yake
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Plagiarism Detector — Enhanced", layout="wide")

# --- SSL & NLTK setup ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)

# --- TextRazor setup ---
TEXTRAZOR_API_KEY = os.getenv("TEXTRAZOR_API_KEY")
textrazor.api_key = TEXTRAZOR_API_KEY

# --- Sentence-BERT model (cached so it loads only once) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- YAKE keyword extractor ---
kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=20)

# ===================== UTILITY FUNCTIONS =====================

def get_sentences(text):
    return tokenize.sent_tokenize(text, language='english')

@st.cache_data(show_spinner=False)
def get_textrazor_results(text):
    """Call TextRazor API with caching to avoid duplicate calls for same text."""
    client = textrazor.TextRazor(extractors=["entities", "topics", "words"])
    response = client.analyze(text)
    urls = []
    similarities = []
    for entity in response.entities():
        if entity.wikipedia_link:
            urls.append(entity.wikipedia_link)
            similarities.append(entity.relevance_score)
    if len(urls) == 0:
        return ["No match found"], [0]
    return urls, similarities

def get_keywords(text):
    return set([kw[0].lower() for kw in kw_extractor.extract_keywords(text)])

@st.cache_data(show_spinner=False)
def get_semantic_similarity(text1, text2):
    """Compute semantic similarity using Sentence-BERT embeddings."""
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

def get_combined_similarity(sent1, sent2, urls_sim):
    """Weighted combination: 50% semantic + 30% keyword overlap + 20% entity relevance."""
    sem_sim = get_semantic_similarity(sent1, sent2)
    kw1 = get_keywords(sent1)
    kw2 = get_keywords(sent2)
    if len(kw1 | kw2) > 0:
        keyword_overlap = len(kw1 & kw2) / len(kw1 | kw2)
    else:
        keyword_overlap = 0
    return round((0.5 * sem_sim + 0.3 * keyword_overlap + 0.2 * urls_sim), 4)

# ===================== FILE READERS =====================

def read_text_file(file):
    return file.read().decode('utf-8')

def read_docx_file(file):
    return docx2txt.process(file)

def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_from_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return read_text_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            return read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx_file(uploaded_file)
    return ""

# ===================== HIGHLIGHTED TEXT VIEW =====================

def render_highlighted_text(sentences, scores, threshold=0.4):
    """Show sentences color-coded by plagiarism risk: red (high), orange (medium), green (low)."""
    st.write("### Highlighted Text View")
    html_parts = []
    for sent, score in zip(sentences, scores):
        if score >= threshold:
            color = "#ff4d4d"  # red — high plagiarism risk
            label = "High"
        elif score >= threshold * 0.6:
            color = "#ffa500"  # orange — medium
            label = "Medium"
        else:
            color = "#4caf50"  # green — low
            label = "Low"
        escaped = sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f'<span style="background-color:{color};color:white;padding:2px 6px;'
            f'border-radius:4px;margin:2px;display:inline" '
            f'title="Score: {score:.2f} ({label})">{escaped}</span> '
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    st.caption("🔴 High risk  🟠 Medium risk  🟢 Low risk — hover over a sentence to see its score")

# ===================== CHARTS =====================

def plot_all_charts(df, score_col):
    col1, col2 = st.columns(2)
    with col1:
        plot_bar(df, score_col)
    with col2:
        plot_scatter(df, score_col)
    plot_heatmap(df, score_col)

def plot_scatter(df, score_col):
    fig = px.scatter(
        df, x=df.index, y=score_col,
        color=score_col,
        color_continuous_scale="RdYlGn_r",
        labels={"index": "Sentence #", score_col: "Similarity"},
        title="Similarity Scatter Plot"
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df, score_col):
    fig = px.bar(
        df, x=df.index, y=score_col,
        color=score_col,
        color_continuous_scale="RdYlGn_r",
        labels={"index": "Sentence #", score_col: "Similarity"},
        title="Similarity Bar Chart"
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap(df, score_col):
    """Single-row heatmap showing sentence-level similarity at a glance."""
    fig = go.Figure(data=go.Heatmap(
        z=[df[score_col].tolist()],
        x=[f"S{i+1}" for i in range(len(df))],
        y=["Similarity"],
        colorscale="RdYlGn_r",
        text=[[f"{v:.2f}" for v in df[score_col]]],
        texttemplate="%{text}",
        hovertemplate="Sentence %{x}<br>Score: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(title="Sentence Similarity Heatmap", height=200, margin=dict(l=10, r=10, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ===================== STREAMLIT UI =====================

st.title("🔍 Plagiarism Detector (Enhanced)")
st.caption("Powered by TextRazor + Sentence-BERT Embeddings + YAKE Keywords")

detection_option = st.radio(
    "Choose detection method:",
    ("Web-based plagiarism check", "File-based plagiarism check (Source vs Derived)")
)

# -------------------- MODE A: WEB-BASED --------------------
if detection_option == "Web-based plagiarism check":
    st.subheader("Enter text or upload a file")
    input_mode = st.radio("Input method:", ("Enter text", "Upload file"))

    if input_mode == "Enter text":
        text = st.text_area("Enter text here", height=200)
    else:
        uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
        text = get_text_from_file(uploaded_file) if uploaded_file else ""

    if st.button("Check for plagiarism"):
        if not text.strip():
            st.warning("No text found to check.")
            st.stop()

        sentences = get_sentences(text)
        progress = st.progress(0, text="Analyzing sentences...")
        combined_scores = []
        urls_list = []
        entity_sims = []
        sentences_list = []

        for idx, sentence in enumerate(sentences):
            urls, url_similarities = get_textrazor_results(sentence)
            for i, url in enumerate(urls):
                url_sim = url_similarities[i]
                comb_sim = get_combined_similarity(sentence, sentence, url_sim)
                sentences_list.append(sentence)
                urls_list.append(url)
                entity_sims.append(url_sim)
                combined_scores.append(comb_sim)
            progress.progress((idx + 1) / len(sentences), text=f"Analyzed {idx + 1}/{len(sentences)} sentences")

        progress.empty()

        df = pd.DataFrame({
            "Sentence": sentences_list,
            "URL": urls_list,
            "EntityRelevance": entity_sims,
            "CombinedSimilarity": combined_scores
        }).sort_values(by="CombinedSimilarity", ascending=False).reset_index(drop=True)

        if df.empty:
            st.success("No plagiarism detected!")
        else:
            avg_score = sum(combined_scores) / len(combined_scores)
            st.metric("Overall Plagiarism Score", f"{avg_score * 100:.2f}%")

            # Highlighted text — one score per sentence (max across its URLs)
            sentence_max_scores = df.groupby("Sentence", sort=False)["CombinedSimilarity"].max()
            ordered_scores = [sentence_max_scores.get(s, 0) for s in sentences]
            render_highlighted_text(sentences, ordered_scores)

            st.write("### Detailed Results")
            st.dataframe(df, use_container_width=True)

            # Charts use per-sentence max for clarity
            chart_df = df.groupby("Sentence", sort=False).agg(
                CombinedSimilarity=("CombinedSimilarity", "max"),
                URL=("URL", "first")
            ).reset_index()
            plot_all_charts(chart_df, "CombinedSimilarity")

# -------------------- MODE B: FILE-BASED --------------------
elif detection_option == "File-based plagiarism check (Source vs Derived)":
    st.subheader("Upload two files for comparison")
    col_src, col_der = st.columns(2)
    with col_src:
        source_file = st.file_uploader("Source File", type=["docx", "pdf", "txt"], key="src")
    with col_der:
        derived_file = st.file_uploader("Derived File", type=["docx", "pdf", "txt"], key="der")

    if st.button("Check for plagiarism"):
        if not source_file or not derived_file:
            st.warning("Both files must be uploaded.")
            st.stop()

        source_text = get_text_from_file(source_file)
        derived_text = get_text_from_file(derived_file)
        if not source_text.strip() or not derived_text.strip():
            st.warning("One or both files have no readable content.")
            st.stop()

        # --- Overall similarity ---
        sem_score = get_semantic_similarity(source_text, derived_text)
        src_kw = get_keywords(source_text)
        der_kw = get_keywords(derived_text)
        kw_union = src_kw | der_kw
        keyword_sim = len(src_kw & der_kw) / len(kw_union) if len(kw_union) > 0 else 0
        overall = round(0.7 * sem_score + 0.3 * keyword_sim, 4)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Overall Similarity", f"{overall * 100:.2f}%")
        col_m2.metric("Semantic Similarity", f"{sem_score * 100:.2f}%")
        col_m3.metric("Keyword Overlap", f"{keyword_sim * 100:.2f}%")

        # --- Sentence-level alignment ---
        derived_sentences = get_sentences(derived_text)
        source_sentences = get_sentences(source_text)

        progress = st.progress(0, text="Matching sentences...")
        best_matches = []
        best_match_sources = []

        for idx, sent in enumerate(derived_sentences):
            best_sim = 0
            best_src = ""
            for src_sent in source_sentences:
                sim = get_combined_similarity(sent, src_sent, 0.5)
                if sim > best_sim:
                    best_sim = sim
                    best_src = src_sent
            best_matches.append(best_sim)
            best_match_sources.append(best_src)
            progress.progress((idx + 1) / len(derived_sentences),
                              text=f"Matched {idx + 1}/{len(derived_sentences)} sentences")

        progress.empty()

        df = pd.DataFrame({
            "Derived Sentence": derived_sentences,
            "Best Matching Source Sentence": best_match_sources,
            "CombinedSimilarity": best_matches
        }).sort_values(by="CombinedSimilarity", ascending=False).reset_index(drop=True)

        # Highlighted text view
        render_highlighted_text(derived_sentences, best_matches)

        st.write("### Sentence-Level Alignment")
        st.dataframe(df, use_container_width=True)

        # Charts
        chart_df = df[["Derived Sentence", "CombinedSimilarity"]].rename(
            columns={"Derived Sentence": "Sentence"}
        ).reset_index(drop=True)
        plot_all_charts(chart_df, "CombinedSimilarity")
