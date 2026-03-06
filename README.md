# Plagiarism Checker

A web-based plagiarism detection tool built with **Streamlit** and **TextRazor NLP API**. It analyzes text by extracting named entities, matching them to known web sources (Wikipedia), and computing relevance-based similarity scores.

## Features

- **Web-based Plagiarism Check** — Paste text or upload a file; each sentence is analyzed against web sources via TextRazor's entity extraction.
- **File-based Plagiarism Check** — Upload a source and a derived file to compute direct cosine similarity between them, plus sentence-level web analysis.
- **Multi-format Support** — Accepts `.txt`, `.pdf`, and `.docx` files.
- **Interactive Visualizations** — Scatter, bar, and pie charts powered by Plotly.
- **Detailed Results Table** — Per-sentence breakdown with matched URLs and similarity scores.

## System Architecture

```
User Input (Text / PDF / DOCX / TXT)
         │
         ▼
┌─────────────────────────┐
│    File Parsing Layer    │
│  PyPDF2 · docx2txt · io │
└────────────┬────────────┘
             │ Raw Text
             ▼
┌─────────────────────────┐
│   NLTK Sentence Tokenizer  │
│   sent_tokenize()          │
└────────────┬────────────┘
             │ [sentence₁, sentence₂, ...]
             ▼
┌──────────────────────────────┐
│     TextRazor API            │
│  Entity & Topic Extraction   │
│  → Wikipedia URLs            │
│  → Relevance Scores          │
└────────────┬─────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
 Web-based        File-based
 Mode             Mode
 (entity          (CountVectorizer +
  relevance)       Cosine Similarity)
     │                │
     ▼                ▼
┌──────────────────────────────┐
│   Streamlit Dashboard        │
│   Results Table + Plotly     │
│   Scatter · Bar · Pie Charts │
└──────────────────────────────┘
```

## Data Flow

### Mode 1: Web-based Plagiarism Check

1. User enters text or uploads a file.
2. Text is extracted and split into sentences using NLTK.
3. Each sentence is sent to the **TextRazor API**, which returns:
   - Named entities with **Wikipedia links**
   - **Relevance scores** (0–1) indicating how strongly the entity relates to the sentence.
4. Results are aggregated into a DataFrame and the **average relevance score** is reported as the plagiarism percentage.
5. Interactive charts visualize the per-sentence similarity distribution.

### Mode 2: File-based Plagiarism Check (Source vs Derived)

1. User uploads two files — a source document and a derived document.
2. Both are parsed into raw text.
3. **Overall similarity** is computed using `CountVectorizer` (bag-of-words) + `cosine_similarity` from scikit-learn.
4. The derived text is additionally sentence-tokenized and each sentence is analyzed through TextRazor for web-based entity matching.
5. Results are displayed with an overall plagiarism percentage and a detailed breakdown.

## Tech Stack

| Component | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI framework |
| [TextRazor](https://www.textrazor.com/) | NLP API — entity extraction, topic detection |
| [NLTK](https://www.nltk.org/) | Sentence tokenization |
| [scikit-learn](https://scikit-learn.org/) | CountVectorizer + cosine similarity |
| [PyPDF2](https://pypdf2.readthedocs.io/) | PDF text extraction |
| [docx2txt](https://github.com/ankushshah89/python-docx2txt) | Word document text extraction |
| [Plotly](https://plotly.com/python/) | Interactive charts |

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/BHARGAV-CHINTAPATLA/plagiarism_checker.git
cd plagiarism_checker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the project root:

```
TEXTRAZOR_API_KEY=your_textrazor_api_key_here
```

Get a free API key at [textrazor.com](https://www.textrazor.com/).

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## Project Structure

```
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env                # API key (not tracked by git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Screenshots

After launching the app, you will see:

- A **radio button** to choose between web-based or file-based plagiarism detection.
- **Text input area** or **file uploader** depending on the selected mode.
- A **results table** showing each sentence, matched URLs, and similarity scores.
- **Scatter, Bar, and Pie charts** for visual analysis.

## License

This project is for educational purposes.
