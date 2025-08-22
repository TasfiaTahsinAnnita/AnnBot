import os
import io
import requests
import pdfplumber
import streamlit as st

from typing import List

# LangChain core
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings & Vector store (community package)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google Generative AI (Gemini) LLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Conversational QA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# -----------------------------
# Streamlit page setup & styles
# -----------------------------
st.set_page_config(page_title="📚 PDF Question Answering Bot", layout="wide")

st.markdown(
    """
    <style>
        .reportview-container { background-color: #f4f4f9; color: #333; }
        .sidebar .sidebar-content { background-color: #f0f0f5; }
        h1 { color: #004c8c; }
        .stButton>button { background-color: #0066ff; color: white; }
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
# Helpers
# -----------------------------
def get_api_key() -> str:
    """
    Read Google API key from Streamlit secrets or environment.
    """
    key = st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error(
            "Missing Google API key. Please set `GOOGLE_API_KEY` in Streamlit secrets "
            "or as an environment variable."
        )
        st.stop()
    return key


def download_pdf_from_github(pdf_url: str) -> bytes:
    """
    Download a PDF from a GitHub raw URL and return bytes.
    """
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    return resp.content


def extract_page_docs_from_pdf_bytes(pdf_bytes: bytes) -> List[Document]:
    """
    Extract page-wise text as LangChain Documents with page metadata.
    """
    docs = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


@st.cache_data(show_spinner=False)
def split_into_chunks(page_docs: List[Document]) -> List[Document]:
    """
    Split page documents into overlapping chunks while preserving page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(page_docs)


@st.cache_resource(show_spinner=False)
def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector index from chunked documents with HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embedding=embeddings)
    return vs


def build_qa_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """
    Create a ConversationalRetrievalChain with memory and a strict prompt
    that answers only from the provided context.
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
        template=prompt_template, input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )
    return qa


def format_sources(source_documents: List[Document]) -> str:
    """
    Render source page numbers and short snippets.
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
        # Add a tiny, safe snippet (first ~220 chars)
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if snippet:
            snippets.append(f'<div class="snippet">{snippet[:220]}{"..." if len(snippet)>220 else ""}</div>')
    chips_html = " ".join(sorted(chips, key=lambda x: int(x.split("Page ")[1].split("<")[0])))
    snippets_html = "\n".join(snippets[:3])  # show up to 3 short snippets
    return f"{chips_html}<div style='margin-top:6px'>{snippets_html}</div>"


# -----------------------------
# UI
# -----------------------------
st.title("📚 PDF Question Answering Bot")

st.sidebar.header("About This App")
st.sidebar.write(
    "Ask questions about the financial policy PDF. The bot retrieves relevant parts of the "
    "document using vector search, remembers your conversation, and cites the page numbers."
)

# PDF source: default to your GitHub raw URL, but allow override.
default_pdf_url = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"
pdf_url = st.sidebar.text_input("PDF URL (raw .pdf)", value=default_pdf_url)

# API key setup
GOOGLE_API_KEY = get_api_key()
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # for langchain_google_genai

# Load & process PDF
with st.spinner("Downloading and processing the PDF..."):
    try:
        pdf_bytes = download_pdf_from_github(pdf_url)
    except Exception as e:
        st.error(f"Failed to download PDF: {e}")
        st.stop()

    page_docs = extract_page_docs_from_pdf_bytes(pdf_bytes)
    if not page_docs:
        st.error("Could not extract any text from the PDF.")
        st.stop()

    chunks = split_into_chunks(page_docs)
    vector_store = build_vector_store(chunks)
    qa_chain = build_qa_chain(vector_store)

st.success("PDF loaded successfully! You can now ask questions.")

col1, col2 = st.columns(2)

with col1:
    st.header("PDF Preview (first page text)")
    st.text_area(
        "Sample Text",
        value=(page_docs[0].page_content[:2000] + ("..." if len(page_docs[0].page_content) > 2000 else "")),
        height=300,
    )
    st.caption("Note: The model searches the entire PDF, not just this preview.")

with col2:
    st.header("Ask a Question")
    question = st.text_input("Enter your question (e.g., 'What are the short-term objectives?')")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Submit", type="primary") and question:
        with st.spinner("Thinking..."):
            result = qa_chain({"question": question})
            answer = result.get("answer", "").strip()
            sources = result.get("source_documents", [])

        st.write("**Bot response:**")
        st.write(answer if answer else "_Answer is not available in the text._")

        # Citations (page numbers + small snippets)
        st.markdown("**Sources:**", unsafe_allow_html=True)
        st.markdown(format_sources(sources), unsafe_allow_html=True)

        # Keep a running chat log (optional display)
        st.session_state.history.append((question, answer))

    if st.session_state.history:
        st.markdown("---")
        st.subheader("Conversation Memory")
        for i, (q, a) in enumerate(st.session_state.history[-6:], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")


# Footer tip
st.caption(
    "Tip: Ask a follow-up like 'What about debt?' — the bot remembers the context."
)
