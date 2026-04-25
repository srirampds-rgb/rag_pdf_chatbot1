"""
app.py — Main Streamlit application for PDF Question-Answering Chatbot using RAG.

Architecture:
  - Streamlit handles the UI and session state
  - LangChain orchestrates the RAG pipeline
  - FAISS stores and retrieves vector embeddings
  - OpenAI provides both embeddings and chat completions
"""

import os
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Constants / paths ─────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join("data", "uploads")
FAISS_DIR  = os.path.join("data", "faiss_indexes")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR,  exist_ok=True)

# ── System prompt (grounded, no hallucination) ────────────────────────────────
SYSTEM_PROMPT = """You are a PDF question-answering assistant.

Rules:
- Answer ONLY from the retrieved context taken from the uploaded PDF.
- If the answer is not clearly supported by the retrieved context, say exactly:
  "I could not find enough information in the uploaded PDF to answer that confidently."
- Do NOT invent or infer facts beyond what the context explicitly states.
- Keep answers clear, direct, and helpful.
- When relevant, mention that the answer is based on the uploaded document."""

# ── User input template ───────────────────────────────────────────────────────
USER_TEMPLATE = """Question: {question}

Retrieved context:
{context}

Instructions:
Answer the question only using the retrieved context above.
If the context does not contain enough information, say:
"I could not find enough information in the uploaded PDF to answer that confidently."
"""

FALLBACK_ANSWER = (
    "I could not find enough information in the uploaded PDF to answer that confidently."
)

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    """Return the OpenAI API key; raise a clear error if missing."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file and restart."
        )
    return key


def file_hash(path: str) -> str:
    """Return an MD5 hash of a file's contents for cache-key purposes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Persist the Streamlit UploadedFile to disk under UPLOAD_DIR.
    Returns the absolute path to the saved file.
    """
    dest = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def load_and_split_pdf(pdf_path: str) -> list:
    """
    Load a PDF with PyPDFLoader and split it into overlapping chunks.
    Raises ValueError if no text can be extracted (e.g. scanned / empty PDF).
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

    # Filter out pages with no meaningful text
    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned image-only PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(pages)


def build_or_load_faiss(chunks: list, index_path: str, embeddings) -> FAISS:
    """
    If a FAISS index already exists at index_path, load it.
    Otherwise build a new one from chunks and save it.
    """
    if os.path.exists(index_path):
        try:
            db = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            return db, False   # False = was loaded, not rebuilt
        except Exception:
            pass  # Fall through and rebuild if loading fails

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)
    return db, True   # True = was newly built


def retrieve_context(db: FAISS, question: str, k: int = 4) -> str:
    """
    Run a similarity search and return concatenated chunk text.
    Returns an empty string if retrieval fails or returns nothing useful.
    """
    try:
        docs = db.similarity_search(question, k=k)
        if not docs:
            return ""
        return "\n\n---\n\n".join(d.page_content for d in docs)
    except Exception:
        return ""


def ask_llm(question: str, context: str, api_key: str) -> str:
    """
    Send the question + retrieved context to ChatGPT and return the answer.
    Falls back gracefully on any API error.
    """
    if not context.strip():
        return FALLBACK_ANSWER

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key,
    )

    prompt = USER_TEMPLATE.format(question=question, context=context)

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        answer   = response.content.strip()
        return answer if answer else FALLBACK_ANSWER
    except Exception as e:
        return f"⚠️ LLM API error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        page_icon="📄",
        layout="wide",
    )

    # ── Custom CSS for a clean chat look ──────────────────────────────────────
    st.markdown("""
    <style>
        .stChatMessage { border-radius: 12px; padding: 8px; }
        .success-box {
            background: #d4edda; border: 1px solid #28a745;
            border-radius: 8px; padding: 10px 14px; color: #155724;
            font-weight: 500; margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📄 PDF Question-Answering Chatbot")
    st.caption("Upload a PDF and ask questions — answers come exclusively from your document.")

    # ── Session state initialisation ──────────────────────────────────────────
    if "chat_history"  not in st.session_state:
        st.session_state.chat_history  = []   # list of {"role": ..., "content": ...}
    if "vector_db"     not in st.session_state:
        st.session_state.vector_db     = None
    if "current_file"  not in st.session_state:
        st.session_state.current_file  = None

    # ── Sidebar — PDF upload ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 Upload PDF")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded:
            # Only re-process if the file changed
            if st.session_state.current_file != uploaded.name:
                with st.spinner("Saving PDF to disk…"):
                    try:
                        pdf_path = save_uploaded_pdf(uploaded)
                        st.markdown(
                            '<div class="success-box">✅ PDF uploaded and saved successfully!</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"❌ Could not save PDF: {e}")
                        st.stop()

                # Build/load FAISS index
                with st.spinner("Building vector database…"):
                    try:
                        api_key    = get_api_key()
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

                        file_id    = file_hash(pdf_path)
                        index_path = os.path.join(FAISS_DIR, file_id)

                        chunks = load_and_split_pdf(pdf_path)
                        db, rebuilt = build_or_load_faiss(chunks, index_path, embeddings)

                        st.session_state.vector_db    = db
                        st.session_state.current_file = uploaded.name
                        st.session_state.chat_history = []   # reset chat on new file

                        msg = (
                            "✅ Vector database created successfully!"
                            if rebuilt
                            else "✅ Vector database loaded from cache!"
                        )
                        st.markdown(
                            f'<div class="success-box">{msg}</div>',
                            unsafe_allow_html=True,
                        )

                    except EnvironmentError as e:
                        st.error(f"🔑 API Key Error: {e}")
                        st.stop()
                    except ValueError as e:
                        st.error(f"📄 PDF Error: {e}")
                        st.stop()
                    except RuntimeError as e:
                        st.error(f"❌ Processing Error: {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                        st.stop()
            else:
                st.success(f"✅ **{uploaded.name}** is ready. Ask your questions!")

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        st.caption("Powered by LangChain · FAISS · OpenAI")

    # ── Main area — chat interface ────────────────────────────────────────────
    if st.session_state.vector_db is None:
        st.info("👈 Upload a PDF from the sidebar to get started.")
        return

    # Render existing chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Accept new user question
    if question := st.chat_input("Ask a question about your PDF…"):
        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching PDF and generating answer…"):
                try:
                    api_key = get_api_key()
                    context = retrieve_context(st.session_state.vector_db, question)
                    answer  = ask_llm(question, context, api_key)
                except EnvironmentError as e:
                    answer = f"🔑 API Key Error: {e}"
                except Exception as e:
                    answer = f"⚠️ Error generating answer: {e}"

            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
