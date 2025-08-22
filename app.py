import os
import io
import requests
import pdfplumber  # Updated for better PDF text extraction
import streamlit as st

from typing import List, Tuple, Dict, Any
from hashlib import md5

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Embeddings & Vector store (community versions preserve metadata nicely)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google Generative AI (Gemini) LLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Conversational QA with memory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# -----------------------------
# Google API Key Setup (hard-coded per your preference)
# -----------------------------
GOOGLE_API_KEY = "AIzaSyDxd5WI0-AcioR9KqjESVElivPzdk-QLo8"
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # used by langchain_google_genai


# -----------------------------
# Custom CSS for styling
# -----------------------------
st.set_page_config(page_title="PDF Question Answering Bot", layout="wide")

st.markdown(
    """
    <style>
        .reportview-container {
            background-color: #f4f4f9;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f5;
        }
        h1 {
            color: #004c8c;
        }
        .stButton>button {
            background-color: #0066ff;
            color: white;
        }
        .source-chip {
            display: inline-block; padding: 2px 8px; margin: 2px 6px 0 0;
            border-radius: 12px; background: #eef2ff; font-size: 12px;
        }
        .snippet {
            background: #fafafa; border-left: 3px solid #c7d2fe;
            padding: 8px 10px; margin: 6px 0; font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# PDF helpers
# -----------------------------
def download_pdf_from_github(pdf_url: str) -> bytes:
    """Download PDF bytes from a raw GitHub URL."""
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    return resp.content


def extract_page_docs_from_pdf_bytes(pdf_bytes: bytes) -> List[Document]:
    """
    Extract text page-by-page and keep page numbers in metadata,
    so we can cite Page X in answers.
    """
    docs = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


# ---------- Caching helpers (hashable inputs only) ----------
def _serialize_docs(docs: List[Document]) -> List[Tuple[str, Dict[str, Any]]]:
    """Turn Document objects into hashable tuples (content, metadata)."""
    out = []
    for d in docs:
        # metadata is dict[str, Any]; convert to sorted tuple for hashing
        meta_sorted = tuple(sorted((k, str(v)) for k, v in (d.metadata or {}).items()))
        out.append((d.page_content, dict(meta_sorted)))
    return out


def _deserialize_docs(serial: List[Tuple[str, Dict[str, Any]]]) -> List[Document]:
    """Rebuild Document objects from serialized tuples."""
    return [Document(page_content=txt, metadata=meta) for txt, meta in serial]


@st.cache_data(show_spinner=False)
def split_into_chunks_cached(pdf_hash: str, page_docs_serial: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Cache-friendly chunking. Inputs are only simple/hashable types.
    Returns serialized chunks as (text, metadata) tuples.
    """
    page_docs = _deserialize_docs(page_docs_serial)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(page_docs)
    return _serialize_docs(chunks)


@st.cache_resource(show_spinner=False)
def build_vector_store_cached(pdf_hash: str, chunks_serial: List[Tuple[str, Dict[str, Any]]]) -> FAISS:
    """
    Cache the FAISS index per PDF hash and chunk content (both hashable).
    """
    chunks = _deserialize_docs(chunks_serial)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embedding=embeddings)
    return vs


def build_conversational_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """
    Conversational chain that:
    - remembers chat history (assignment requirement),
    - answers ONLY from retrieved context,
    - returns source documents for page citations.
    """
    prompt_template = """
Answer the user's question using ONLY the provided context. Be concise and specific.
If the answer is not present in the context, say: "Answer is not available in the text."

Context:
{context}

Question:
{question}

Helpful answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def render_sources(source_documents: List[Document]) -> str:
    """
    Render page-number chips and short snippets for transparency.
    """
    if not source_documents:
        return ""
    seen = set()
    chips = []
    snippets = []
    for d in source_documents:
        page = d.metadata.get("page")
        if page and page not in seen:
            seen.add(page)
            chips.append(f'<span class="source-chip">Page {page}</span>')
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if snippet:
            snippets.append(
                f'<div class="snippet">{snippet[:220]}{"..." if len(snippet) > 220 else ""}</div>'
            )
    chips_html = " ".join(sorted(chips, key=lambda x: int(x.split("Page ")[1].split("<")[0])))
    snippets_html = "\n".join(snippets[:3])  # limit to three short snippets
    return f"{chips_html}<div style='margin-top:6px'>{snippets_html}</div>"


# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("📚 PDF Question Answering Bot")

st.sidebar.header("About This App")
st.sidebar.write(
    "This bot answers your questions based on the content of a PDF loaded from GitHub. "
    "It uses vector search, remembers the conversation, and cites page numbers."
)

# GitHub PDF URL (your original)
pdf_url = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"

# Download the PDF
with st.spinner('Processing the PDF...'):
    try:
        pdf_bytes = download_pdf_from_github(pdf_url)
    except Exception as e:
        st.error(f"Failed to download PDF: {e}")
        st.stop()

    # Make a stable hash key for cache
    pdf_hash = md5(pdf_bytes).hexdigest()

    # Extract and serialize page docs (hashable)
    page_docs = extract_page_docs_from_pdf_bytes(pdf_bytes)
    page_docs_serial = _serialize_docs(page_docs)

    # Split into chunks (cached, hashable)
    chunks_serial = split_into_chunks_cached(pdf_hash, page_docs_serial)

    # Build vector store (cached, hashable inputs)
    vector_store = build_vector_store_cached(pdf_hash, chunks_serial)

    # Build the conversational chain (not cached; fast)
    convo_chain = build_conversational_chain(vector_store)

st.success('PDF loaded successfully! You can now ask questions.')

# Layout with columns: Left column for PDF text and right column for the chatbot
col1, col2 = st.columns(2)

with col1:
    st.header("PDF Content (first page preview)")
    preview = page_docs[0].page_content if page_docs else ""
    st.text_area("Preview Text", value=preview[:2000], height=300)
    st.caption("Note: The bot searches the entire PDF, not just this preview.")

with col2:
    st.header("Ask a Question")
    question = st.text_input("Enter your question:")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button('Submit'):
        if question:
            with st.spinner("Thinking..."):
                result = convo_chain({"question": question})
                answer = result.get("answer", "").strip()
                sources = result.get("source_documents", [])

            st.write("**Bot response:**")
            st.write(answer if answer else "_Answer is not available in the text._")

            st.markdown("**Sources:**", unsafe_allow_html=True)
            st.markdown(render_sources(sources), unsafe_allow_html=True)

            st.session_state.history.append((question, answer))
        else:
            st.write("Please enter a question to get an answer.")

    # Show recent conversation (memory)
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Conversation Memory")
        for i, (q, a) in enumerate(st.session_state.history[-6:], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
