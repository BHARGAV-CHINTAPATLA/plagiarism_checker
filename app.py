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
from dotenv import load_dotenv

load_dotenv()

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
TEXTRAZOR_API_KEY = os.getenv("TEXTRAZOR_API_KEY")
textrazor.api_key = TEXTRAZOR_API_KEY
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
def get_similarity(text1, text2):
    text_list = [text1, text2]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    return cosine_similarity(count_matrix)[0][1]
def plot_scatter(df):
    fig = px.scatter(df, x='Sentence', y='Similarity', color='Similarity', title='Similarity Scatter Plot')
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=40, b=40),
        height=400,
        width=800,
    )
    st.plotly_chart(fig, use_container_width=True)
def plot_bar(df):
    fig = px.bar(df, x='Sentence', y='Similarity', color='URL', title='Similarity Bar Chart')
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=40, b=40),
        height=400,
        width=800,
    )
    st.plotly_chart(fig, use_container_width=True)
def plot_pie(df):
    fig = px.pie(df, values='Similarity', names='Sentence', title='Similarity Pie Chart')
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=40, b=40),
        height=400,
        width=800,
    )
    st.plotly_chart(fig, use_container_width=True)
st.set_page_config(page_title='Plagiarism Detection with TextRazor')
st.title('Plagiarism Detector (TextRazor API)')
st.write("""
### Choose your plagiarism detection method:
""")
detection_option = st.radio(
    "Select your plagiarism detection method:",
    ('Web-based plagiarism check', 'File-based plagiarism check (Source vs Derived)')
)
if detection_option == 'Web-based plagiarism check':
    st.write("""
    ### Enter the text or upload a file to check for plagiarism
    """)
    option = st.radio(
        "Select input option:",
        ('Enter text', 'Upload file')
    )
    if option == 'Enter text':
        text = st.text_area("Enter text here", height=200)
        uploaded_files = []
    elif option == 'Upload file':
        uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
        if uploaded_file is not None:
            text = get_text_from_file(uploaded_file)
            uploaded_files = [uploaded_file]
        else:
            text = ""
            uploaded_files = []
    if st.button('Check for plagiarism'):
        st.write("### Checking for plagiarism...")
        if not text:
            st.write("### No text found for plagiarism check.")
            st.stop()
        sentences = get_sentences(text)
        urls_list = []
        similarities_list = []
        sentences_list = []
        for sentence in sentences:
            urls, similarities = get_textrazor_results(sentence)
            urls_list.extend(urls)
            similarities_list.extend(similarities)
            sentences_list.extend([sentence] * len(urls))
        df = pd.DataFrame({
            "Sentence": sentences_list,
            "URL": urls_list,
            "Similarity": similarities_list
        })
        df = df.sort_values(by=['Similarity'], ascending=False).reset_index(drop=True)
        if len(similarities_list) > 0:
            avg_similarity = sum(similarities_list) / len(similarities_list)
            plagiarism_percentage = avg_similarity * 100
            st.write(f"### Overall Plagiarism Percentage: {plagiarism_percentage:.2f}%")
        if df.empty:
            st.write("### No plagiarism detected!")
        else:
            st.write("### Plagiarism Results:")
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
            plot_scatter(df)
            plot_bar(df)
            plot_pie(df)
elif detection_option == 'File-based plagiarism check (Source vs Derived)':
    st.write("""
    ### Upload two files for plagiarism check (Source vs Derived Content)
    """)
    source_file = st.file_uploader("Upload Source File (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], key="source_file")
    derived_file = st.file_uploader("Upload Derived File (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], key="derived_file")
    if st.button('Check for plagiarism'):
        st.write("### Checking for plagiarism...")
        if source_file is not None and derived_file is not None:
            source_text = get_text_from_file(source_file)
            derived_text = get_text_from_file(derived_file)
            if not source_text or not derived_text:
                st.write("### One or both files do not contain any readable content!")
                st.stop()
            similarity_score = get_similarity(source_text, derived_text)
            plagiarism_percentage = similarity_score * 100
            st.write(f"### Similarity Score: {similarity_score:.2f}")
            st.write(f"### Plagiarism Percentage: {plagiarism_percentage:.2f}%")
            sentences = get_sentences(derived_text)
            urls_list = []
            similarities_list = []
            sentences_list = []
            for sentence in sentences:
                urls, similarities = get_textrazor_results(sentence)
                urls_list.extend(urls)
                similarities_list.extend(similarities)
                sentences_list.extend([sentence] * len(urls))
            df = pd.DataFrame({
                "Sentence": sentences_list,
                "URL": urls_list,
                "Similarity": similarities_list
            })
            df = df.sort_values(by=['Similarity'], ascending=False).reset_index(drop=True)
            if df.empty:
                st.write("### No plagiarism detected!")
            else:
                st.write("### Plagiarism Results:")
                st.write(df.to_html(escape=False), unsafe_allow_html=True)
                plot_scatter(df)
                plot_bar(df)
                plot_pie(df)