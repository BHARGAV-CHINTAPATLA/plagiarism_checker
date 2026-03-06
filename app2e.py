import streamlit as st
import pandas as pd
import nltk
import os
import ssl
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px
import textrazor
from sentence_transformers import SentenceTransformer, util
import yake
from dotenv import load_dotenv

load_dotenv()

# Handle SSL and NLTK
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

# TextRazor setup
TEXTRAZOR_API_KEY = os.getenv("TEXTRAZOR_API_KEY")
textrazor.api_key = TEXTRAZOR_API_KEY

# Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# YAKE keyword extractor
kw_extractor = yake.KeywordExtractor()

def get_sentences(text):
    return tokenize.sent_tokenize(text, language='english')

def get_textrazor_results(text):
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

def get_semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

def get_combined_similarity(sent1, sent2, urls_sim):
    sem_sim = get_semantic_similarity(sent1, sent2)
    kw1 = get_keywords(sent1)
    kw2 = get_keywords(sent2)
    if len(kw1 | kw2) > 0:
        keyword_overlap = len(kw1 & kw2) / len(kw1 | kw2)
    else:
        keyword_overlap = 0
    return round((0.5 * sem_sim + 0.3 * keyword_overlap + 0.2 * urls_sim), 2)

def read_text_file(file):
    with io.open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

def read_docx_file(file):
    return docx2txt.process(file)

def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
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

def plot_all_charts(df):
    plot_scatter(df)
    plot_bar(df)
    plot_pie(df)

def plot_scatter(df):
    fig = px.scatter(df, x='Sentence', y='CombinedSimilarity', color='CombinedSimilarity', title='Combined Similarity (Scatter)')
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df):
    fig = px.bar(df, x='Sentence', y='CombinedSimilarity', color='URL', title='Combined Similarity (Bar)')
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df):
    fig = px.pie(df, values='CombinedSimilarity', names='Sentence', title='Similarity (Pie)')
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.set_page_config(page_title='Plagiarism Detector with NLP + TextRazor')
st.title('Plagiarism Detector (TextRazor + Embeddings + Keywords)')

detection_option = st.radio("Choose plagiarism detection method:", 
                            ('Web-based plagiarism check', 'File-based plagiarism check (Source vs Derived)'))

if detection_option == 'Web-based plagiarism check':
    input_mode = st.radio("Select input method:", ('Enter text', 'Upload file'))
    if input_mode == 'Enter text':
        text = st.text_area("Enter text here", height=200)
    else:
        uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
        text = get_text_from_file(uploaded_file) if uploaded_file else ""

    if st.button('Check for plagiarism'):
        if not text.strip():
            st.warning("No text found to check.")
            st.stop()

        st.write("### Analyzing...")
        sentences = get_sentences(text)
        combined_scores = []
        urls_list = []
        similarities_list = []
        sentences_list = []

        for sentence in sentences:
            urls, url_similarities = get_textrazor_results(sentence)
            for i, url in enumerate(urls):
                url_sim = url_similarities[i]
                comb_sim = get_combined_similarity(sentence, sentence, url_sim)  # sentence vs itself
                sentences_list.append(sentence)
                urls_list.append(url)
                similarities_list.append(url_sim)
                combined_scores.append(comb_sim)

        df = pd.DataFrame({
            "Sentence": sentences_list,
            "URL": urls_list,
            "EntitySimilarity": similarities_list,
            "CombinedSimilarity": combined_scores
        }).sort_values(by="CombinedSimilarity", ascending=False).reset_index(drop=True)

        if df.empty:
            st.success("No plagiarism detected.")
        else:
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
            st.write(f"### Average Combined Similarity: {sum(combined_scores)/len(combined_scores)*100:.2f}%")
            plot_all_charts(df)

elif detection_option == 'File-based plagiarism check (Source vs Derived)':
    source_file = st.file_uploader("Upload Source File", type=["docx", "pdf", "txt"], key="src")
    derived_file = st.file_uploader("Upload Derived File", type=["docx", "pdf", "txt"], key="dr")

    if st.button("Check for plagiarism"):
        if not source_file or not derived_file:
            st.warning("Both files must be uploaded.")
            st.stop()

        source_text = get_text_from_file(source_file)
        derived_text = get_text_from_file(derived_file)
        if not source_text.strip() or not derived_text.strip():
            st.warning("One or both files have no readable content.")
            st.stop()

        sim_score = get_semantic_similarity(source_text, derived_text)
        keyword_sim = len(get_keywords(source_text) & get_keywords(derived_text)) / len(get_keywords(source_text) | get_keywords(derived_text))
        combined = round(0.7 * sim_score + 0.3 * keyword_sim, 2)
        st.write(f"### Overall Similarity: {combined*100:.2f}%")

        # Breakdown sentence-level
        derived_sentences = get_sentences(derived_text)
        source_sentences = get_sentences(source_text)
        best_matches = []
        for sent in derived_sentences:
            best_sim = 0
            for src_sent in source_sentences:
                sim = get_combined_similarity(sent, src_sent, 0.5)
                if sim > best_sim:
                    best_sim = sim
            best_matches.append(best_sim)

        df = pd.DataFrame({
            "Sentence": derived_sentences,
            "CombinedSimilarity": best_matches
        }).sort_values(by="CombinedSimilarity", ascending=False).reset_index(drop=True)

        st.write(df.to_html(escape=False), unsafe_allow_html=True)
        plot_all_charts(df)
