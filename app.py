import os
import io
import time
import uuid
import requests
from html import escape
from typing import List

import pdfplumber
import streamlit as st

# Text splitting & documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Community modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google Gemini
import google.generativeai as genai

# Configure API key
GOOGLE_API_KEY = "AIzaSyB6tijdhntntrSl8e5AJE5n1ZpFVdufN_M"
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit page config
st.set_page_config(page_title="What do you want to know?", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        .reportview-container { background-color: #f4f4f9; color: #333; }
        .sidebar .sidebar-content { background-color: #f0f0f5; }
        h1 { color: #004c8c; }
        .stButton>button { background-color: #0066ff; color: white; }

        .chat-meta { font-size: 12px; color: #666; margin: 8px 0; }

        .user-q {
            background: #eef2ff;
            color: #1e40af;
            border: 1px solid #c7d2fe;
            display: inline-block;
            padding: 8px 12px;
            margin: 8px 0 2px 0;
            border-radius: 12px;
            max-width: 900px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .bot-a {
            background: #0b5ed7;
            color: #ffffff;
            display: inline-block;
            padding: 10px 14px;
            margin: 4px 0 12px 18px;
            border-radius: 12px;
            max-width: 900px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .sidebar-chat-title {
            padding: 8px 10px;
            margin: 4px 0;
            border-radius: 8px;
            font-size: 14px;
        }
        .sidebar .stRadio > div { gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# PDF handling
# -----------------------------
def download_pdf_from_github(pdf_url: str) -> bytes:
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    return resp.content

def extract_page_docs_from_pdf_bytes(pdf_bytes: bytes) -> List[Document]:
    docs = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs

def split_into_chunks(page_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(page_docs)

def build_vector_store(chunks: List[Document]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embedding=embeddings)
    return vs

# -----------------------------
# Gemini helper
# -----------------------------
def ask_gemini(context: str, question: str, model_name="gemini-1.5-flash-latest") -> str:
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say: "Answer is not available in the text."

Context:
{context}

Question:
{question}

Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else ""

# -----------------------------
# Load PDF once
# -----------------------------
PDF_URL = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"

if "vector_store" not in st.session_state:
    with st.spinner("Processing the PDF..."):
        pdf_bytes = download_pdf_from_github(PDF_URL)
        page_docs = extract_page_docs_from_pdf_bytes(pdf_bytes)
        if not page_docs:
            st.error("Could not extract any text from the PDF.")
            st.stop()
        chunks = split_into_chunks(page_docs)
        st.session_state.vector_store = build_vector_store(chunks)
    st.success("PDF loaded successfully! You can now ask questions.")

# -----------------------------
# Chat handling
# -----------------------------
def _new_chat(title: str | None = None):
    chat_id = str(uuid.uuid4())
    chat_title = title or f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {
        "title": chat_title,
        "created_at": int(time.time()),
        "history": [],
    }
    st.session_state.nonces[chat_id] = 0
    st.session_state.current_chat_id = chat_id

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "nonces" not in st.session_state:
    st.session_state.nonces = {}

if st.session_state.current_chat_id is None:
    _new_chat("Chat 1")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("AnnBot")
if st.sidebar.button("➕ New Chat", use_container_width=True):
    _new_chat()
    st.rerun()

chat_ids = list(st.session_state.chats.keys())
if st.session_state.current_chat_id not in chat_ids:
    st.session_state.current_chat_id = chat_ids[0]
current_index = chat_ids.index(st.session_state.current_chat_id)

selected_id = st.sidebar.radio(
    label="", options=chat_ids, index=current_index,
    format_func=lambda cid: st.session_state.chats[cid]["title"],
)

if selected_id != st.session_state.current_chat_id:
    st.session_state.current_chat_id = selected_id
    st.rerun()

# -----------------------------
# Main UI
# -----------------------------
st.title("What do you want to know?")
st.subheader(st.session_state.chats[st.session_state.current_chat_id]["title"])
st.caption("Ask questions regarding the Financial policies.")

current_chat_id = st.session_state.current_chat_id
current_chat = st.session_state.chats[current_chat_id]
vector_store = st.session_state.vector_store

# Show chat history
if current_chat["history"]:
    st.markdown("<div class='chat-meta'>Recent messages</div>", unsafe_allow_html=True)
for q, a in current_chat["history"][-20:]:
    st.markdown(f"<div class='user-q'>You: {escape(q)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-a'>Bot: {escape(a)}</div>", unsafe_allow_html=True)

# Input
nonce = st.session_state.nonces.get(current_chat_id, 0)
q_key = f"q_input_{current_chat_id}_{nonce}"
question = st.text_input("Enter your question:", key=q_key)

# Submit
if st.button("Submit", type="primary", key=f"submit_{current_chat_id}"):
    if question.strip():
        with st.spinner("Thinking..."):
            # Retrieve context from FAISS
            docs = vector_store.similarity_search(question.strip(), k=4)
            context = "\n\n".join(d.page_content for d in docs)

            try:
                answer = ask_gemini(context, question.strip())
            except Exception:
                st.error("The model is temporarily unavailable. Please try again.")
                answer = ""

        if answer:
            current_chat["history"].append((question.strip(), answer))

            if len(current_chat["history"]) == 1:
                t = question.strip()
                current_chat["title"] = (t[:30] + "…") if len(t) > 30 else t

            st.session_state.nonces[current_chat_id] += 1
            st.rerun()
    else:
        st.warning("Please ask your queries.")
