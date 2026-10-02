# MarineWise AI

MarineWise AI is a beginner-friendly Streamlit MVP for marine engine troubleshooting and technician training.

## What is fixed in this version

- `GROQ_API_KEY` is read only from Streamlit Secrets or the environment.
- No API key is stored in the repository.
- No ngrok.
- The requested Groq model is `openai/gpt-oss-120b`.
- The Groq Python client is used for the optional web-search fallback.
- The Troubleshooting Agent and Technical Training Agent use the official Groq Python client directly. This avoids the CrewAI/LiteLLM `cache_breakpoint` incompatibility with Groq.
- Manual-first FAISS RAG remains in `st.session_state`.
- Google Drive is not loaded on startup.
- Exact PDF page numbers and source filenames are retained in RAG metadata.
- The app handles missing Tesseract OCR gracefully instead of crashing.
- Dependency versions are pinned.
- The recommended deployment Python version is 3.12.

## Files

```text
MarineWise-AI/
├── app.py
├── agents.py
├── rag.py
├── requirements.txt
└── README.md
```

Also create this local-only file:

```text
.gitignore
```

with:

```text
.venv/
__pycache__/
.streamlit/secrets.toml
*.pyc
```

Do NOT commit `.streamlit/secrets.toml`.

---

# 1. Install Python

Use Python **3.12**.

Check:

```bash
python --version
```

You want:

```text
Python 3.12.x
```

---

# 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 3. Install Python packages

```bash
pip install -r requirements.txt
```

The first time the RAG system is used, Sentence Transformers downloads the small:

```text
all-MiniLM-L6-v2
```

embedding model.

---

# 4. Configure your Groq API key

Never put the key inside `app.py`, `agents.py`, or `rag.py`.

## Local option A — environment variable

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_KEY"
```

macOS/Linux:

```bash
export GROQ_API_KEY="YOUR_GROQ_KEY"
```

## Local option B — Streamlit secrets

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "YOUR_GROQ_KEY"
```

Do not upload that file to GitHub.

---

# 5. Run locally

From the project folder:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

---

# 6. Add manuals

The sidebar lets you:

- upload one or more PDF manuals, OR
- paste a public Google Drive PDF link.

Both are optional.

Google Drive is NOT downloaded when the app starts.

After supplying manuals, click:

```text
Build / Rebuild FAISS Index
```

The index is kept in `st.session_state` during the current Streamlit session.

For best results, use text-based manufacturer manuals. Scanned image-only manuals need OCR before they can be searched by this simple RAG pipeline.

---

# 7. Troubleshooting Agent

Enter:

- Manufacturer
- Engine Model
- Serial Number (optional)
- Defect or Alarm

MarineWise searches the indexed manuals first.

The retrieved records retain:

```text
source filename
PDF page number
chunk similarity
```

If useful manual evidence is not retrieved, the app asks:

```text
Not found in manuals. Do you want me to search online?
```

If you choose online search, the answer is explicitly labeled as web-sourced.

---

# 8. Technical Training

The training page contains:

### 2A Training Material

Enter:

- Engine Model
- Ship
- Training Topic

Choose:

- PDF
- PowerPoint
- Word

The app searches the manuals first and then uses the MarineWise Training Agent to produce the material.

### 2B Quiz Generator

Choose:

- Topic
- MCQ
- Short Question
- True-False
- Number of questions

The generated PDF includes an answer key.

### 2C Score an Assessment

Upload:

- JPG
- JPEG
- PNG

The app uses OCR to read the assessment, then uses the MarineWise agent assessment agent to score it against your supplied answer key.

If the score is below 50%, the app creates a remedial PowerPoint.

---

# 9. OCR note

`pytesseract` is a Python wrapper around the Tesseract OCR program.

Therefore:

## Windows/macOS/Linux local use

Install the Tesseract OCR application separately, then keep:

```text
pytesseract==0.3.13
```

in `requirements.txt`.

If Tesseract is unavailable, MarineWise shows a friendly error instead of crashing.

## Streamlit Community Cloud

Streamlit Community Cloud supports Linux system dependencies through an optional `packages.txt` file.

Because the requested MarineWise MVP consists of five downloadable application files, this repository does not include `packages.txt` by default.

If you want the **Score an Assessment** OCR feature on Streamlit Community Cloud, add this small sixth deployment file at the repository root:

```text
packages.txt
```

containing:

```text
tesseract-ocr
```

This is only a system dependency file; it is not application code.

The rest of MarineWise can deploy without it.

---

# 10. GitHub

Create a repository, for example:

```text
MarineWise-AI
```

Put these five files in the repository:

```text
app.py
agents.py
rag.py
requirements.txt
README.md
```

Also add `.gitignore`:

```text
.venv/
__pycache__/
.streamlit/secrets.toml
*.pyc
```

If you want cloud OCR, also add:

```text
packages.txt
```

with:

```text
tesseract-ocr
```

Then:

```bash
git init
git add app.py agents.py rag.py requirements.txt README.md .gitignore
git commit -m "Initial MarineWise AI MVP"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

If using cloud OCR, include `packages.txt` in the `git add` command.

---

# 11. Streamlit Community Cloud

Open Streamlit Community Cloud and create a new app.

Select:

```text
Repository: YOUR-USERNAME/MarineWise-AI
Branch: main
Main file: app.py
```

For the Python version, select:

```text
3.12
```

In the Secrets field enter:

```toml
GROQ_API_KEY = "YOUR_GROQ_KEY"
```

Then deploy.

Do NOT put the Groq key into GitHub.

---

# 12. First cloud test

After deployment:

1. Open MarineWise AI.
2. Confirm the app loads.
3. Upload a small text-based marine PDF.
4. Click `Build / Rebuild FAISS Index`.
5. Open Troubleshooting Agent.
6. Enter a known alarm or fault from the manual.
7. Check that the answer shows the manual filename and page.
8. Generate a small training PDF.
9. Generate a quiz.
10. Open Learning.
11. If OCR is enabled, upload a clear assessment image and supply an answer key.

---

# Architecture

```text
                  PDF manual
                      |
                      v
                PyMuPDF pages
                      |
                      v
             900-char chunks
             120-char overlap
                      |
                      v
          Sentence Transformer
                      |
                      v
                  FAISS
                      |
             top matching chunks
                      |
                      v
          +---------------------+
          |      MarineWise agent         |
          |                     |
          | Troubleshooting     |
          | Training            |
          | Assessment          |
          +---------------------+
                      |
                      v
             Groq GPT-OSS 120B
                      |
                      v
             MarineWise answer
```

The optional web fallback is separate:

```text
Question not found in manual
            |
            v
User chooses online search
            |
            v
Groq Python client
            |
            v
Groq browser_search
            |
            v
Web-sourced answer
```

## Beginner explanation

### RAG

RAG means Retrieval-Augmented Generation.

Instead of asking the AI to answer from memory:

```text
Question → AI
```

MarineWise does:

```text
Question
   ↓
Search your manuals
   ↓
Find relevant chunks
   ↓
Give those chunks to the AI
   ↓
Answer using the evidence
```

### Advanced RAG

Advanced RAG can add:

- query rewriting
- metadata filters
- reranking
- multiple searches
- hybrid keyword/vector search
- better chunking

This MVP deliberately keeps those pieces simple.

### Chunk

A chunk is a small piece of a manual.

MarineWise currently uses:

```text
Chunk size: 900 characters
Overlap:    120 characters
```

The overlap helps preserve context between neighboring chunks.

### FAISS

FAISS is a vector similarity-search library.

Each chunk is converted into an embedding vector. The user's question is also converted into a vector. FAISS finds the closest chunks.

---

# Important marine safety note

MarineWise AI is an MVP and is not a certified marine engineering system.

Always verify:

- troubleshooting steps
- alarm meanings
- operating limits
- isolation/lockout procedures
- maintenance procedures
- safety requirements

against the current manufacturer documentation and vessel procedures.

The AI should assist a qualified technician, not replace the manufacturer's manual or approved engineering procedures.

### Training Agent workflow

```text
Uploaded/Pasted Manual
        ↓
PDF pages → chunks → embeddings → FAISS
        ↓
Training request (2A / 2B / 2C)
        ↓
Retrieve relevant manual pages
        ↓
MarineWise agent Technical Training Agent
        ↓
Training material / Quiz + Answer Key / Score + Remedial PPT
```

For **2A**, the app also creates a simple learning diagram and offers PDF, PPTX, and DOCX downloads.
For **2B**, the generated quiz includes an Answer Key and can be downloaded as a PDF.
For **2C**, OCR is used for JPG/PNG assessments. If the final score is below 50%, the Training Agent creates a downloadable remedial PowerPoint.
