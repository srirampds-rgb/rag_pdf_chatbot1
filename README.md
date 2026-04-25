# 📄 PDF RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask questions about it. Answers come exclusively from your document — no hallucinations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Vector DB | FAISS (local) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| LLM | OpenAI `gpt-3.5-turbo` |
| PDF parsing | PyPDF |

---

## Folder Structure

```
pdf_rag_chatbot/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .env                    ← Your OpenAI key (NOT committed to git)
├── .env.example            ← Safe template to share
├── README.md               ← This file
│
└── data/
    ├── uploads/            ← Uploaded PDFs are saved here
    └── faiss_indexes/      ← FAISS vector indexes (one per unique PDF)
```

---

## Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Python | 3.9 | `python --version` |
| pip | 23+ | `pip --version` |
| Git (optional) | any | `git --version` |
| OpenAI account | — | platform.openai.com |

---

## Step-by-Step Setup (VS Code)

### 1 — Clone or create the project folder

```bash
# Option A: if you have the files already
cd path/to/pdf_rag_chatbot

# Option B: create fresh
mkdir pdf_rag_chatbot && cd pdf_rag_chatbot
```

### 2 — Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, LangChain, FAISS, OpenAI SDK, and all other required packages.

### 4 — Verify installations

```bash
python -c "import streamlit, langchain, faiss, openai; print('All OK')"
streamlit --version
```

Expected output: `All OK` followed by a Streamlit version string.

### 5 — Create your .env file

```bash
# Copy the example
cp .env.example .env
```

Open `.env` and replace the placeholder with your real OpenAI API key:

```
OPENAI_API_KEY=sk-proj-your-real-key-here
```

> 🔑 Get your key at: https://platform.openai.com/api-keys

### 6 — Create data directories (auto-created on first run, but you can pre-create them)

```bash
mkdir -p data/uploads data/faiss_indexes
```

### 7 — Run the app

```bash
streamlit run app.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Open that URL in your browser.

---

## How to Use

1. **Upload a PDF** using the sidebar uploader.
2. Wait for the two green success messages:
   - ✅ PDF uploaded and saved successfully!
   - ✅ Vector database created successfully!
3. **Type your question** in the chat input at the bottom.
4. Read the grounded answer sourced from your PDF.
5. Ask follow-up questions — full chat history is preserved per session.

---

## How RAG Works (Pipeline)

```
User uploads PDF
      │
      ▼
PyPDFLoader extracts text per page
      │
      ▼
RecursiveCharacterTextSplitter creates overlapping chunks (800 chars / 150 overlap)
      │
      ▼
OpenAI text-embedding-ada-002 converts each chunk → vector
      │
      ▼
FAISS stores vectors locally under data/faiss_indexes/<md5-hash>/
      │
      ▼
User asks question
      │
      ▼
Question is embedded → FAISS similarity search retrieves top-4 chunks
      │
      ▼
Chunks + question sent to gpt-3.5-turbo with grounding system prompt
      │
      ▼
Answer returned (strictly from PDF context)
```

---

## Fallback & Error Handling

| Failure Scenario | What Happens |
|---|---|
| Missing API key | Clear error in sidebar; app stops gracefully |
| Scanned / image-only PDF | `ValueError` caught; user sees helpful message |
| Corrupted PDF | `RuntimeError` caught; user sees helpful message |
| FAISS load failure | Automatically rebuilds the index |
| Empty retrieval results | Returns fallback string (no hallucination) |
| LLM API error | Shows error message with exception detail |

---

## FAISS Index Caching

Each PDF is identified by its **MD5 content hash**. If you re-upload the same file (even with a different name), the existing FAISS index is reused — saving time and OpenAI embedding costs.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'faiss'`
```bash
pip install faiss-cpu
```

### `AuthenticationError` from OpenAI
- Make sure `.env` exists and contains a valid `OPENAI_API_KEY`.
- Ensure you've activated the virtual environment.

### `No extractable text found in this PDF`
- The PDF is likely scanned. Use an OCR tool (e.g. Adobe Acrobat, `ocrmypdf`) to make it text-searchable first.

### Streamlit port already in use
```bash
streamlit run app.py --server.port 8502
```

### VS Code doesn't detect the venv
- Press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `./venv/Scripts/python` (Windows) or `./venv/bin/python` (Mac/Linux).

### `allow_dangerous_deserialization` warning
This is expected when loading a FAISS index you created yourself. The flag is set to `True` safely in `app.py` since the index is locally generated.

---

## Security Notes

- Never commit `.env` to Git. Add it to `.gitignore`:
  ```
  .env
  data/
  venv/
  ```
- The FAISS index contains embeddings derived from your PDF. Keep the `data/` folder private.

---

## Cost Estimate (OpenAI)

| Action | Model | Approx cost |
|---|---|---|
| Embedding a 20-page PDF | text-embedding-ada-002 | ~$0.001 |
| Each Q&A round | gpt-3.5-turbo | ~$0.002 |

GPT-4 can be substituted in `ask_llm()` for higher quality at higher cost.
