import os
import io
import time
import uuid
import requests
import pdfplumber
import streamlit as st

from html import escape
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

GOOGLE_API_KEY = "AIzaSyDxd5WI0-AcioR9KqjESVElivPzdk-QLo8"
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="📚 PDF Question Answering Bot", layout="wide")

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
        .sidebar .stRadio > div { gap: 6px; }
        .sidebar-logo { margin: 6px 0 12px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

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

def build_conversational_chain(vector_store: FAISS, memory: ConversationBufferMemory) -> ConversationalRetrievalChain:
    prompt_template = """
Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say: "Answer is not available in the text."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
        verbose=False,
    )
    return chain

def to_raw_github(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")
    return url

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

def _new_chat(title: str | None = None):
    chat_id = str(uuid.uuid4())
    chat_title = title or f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {
        "title": chat_title,
        "created_at": int(time.time()),
        "history": [],
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
    }
    st.session_state.nonces[chat_id] = 0
    st.session_state.current_chat_id = chat_id

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "nonces" not in st.session_state:
    st.session_state.nonces = {}
if "logo_src" not in st.session_state:
    st.session_state.logo_src = to_raw_github("https://github.com/TasfiaTahsinAnnita/AnnBot/blob/main/Annbotlogo.png")
if "logo_url" not in st.session_state:
    st.session_state.logo_url = st.session_state.logo_src

if st.session_state.current_chat_id is None:
    _new_chat("Chat 1")

st.sidebar.markdown("### ")
logo_container = st.sidebar.container()
if st.session_state.logo_src:
    logo_container.image(st.session_state.logo_src, width=140)
logo_container.markdown("<div class='sidebar-logo'></div>", unsafe_allow_html=True)

with st.sidebar.expander("Branding", expanded=False):
    url = st.text_input("Logo URL (png/jpg)", value=st.session_state.logo_url, key="logo_url_input")
    upl = st.file_uploader("Upload logo", type=["png","jpg","jpeg","webp"], key="logo_uploader")
    if upl is not None:
        st.session_state.logo_src = upl
        st.session_state.logo_url = ""
        st.rerun()
    elif url:
        st.session_state.logo_src = to_raw_github(url)
        st.session_state.logo_url = url
        st.rerun()

st.sidebar.header("Chats")
if st.sidebar.button("➕ New Chat"):
    _new_chat()
    st.rerun()

chat_ids = list(st.session_state.chats.keys())
if st.session_state.current_chat_id not in chat_ids:
    st.session_state.current_chat_id = chat_ids[0]
current_index = chat_ids.index(st.session_state.current_chat_id)

selected_id = st.sidebar.radio(label="", options=chat_ids, index=current_index, format_func=lambda cid: st.session_state.chats[cid]["title"])
if selected_id != st.session_state.current_chat_id:
    st.session_state.current_chat_id = selected_id
    st.rerun()

st.title("📚 PDF Question Answering Bot")
st.subheader(st.session_state.chats[st.session_state.current_chat_id]["title"])
st.caption("Ask questions about the financial policy PDF. Each chat has its own memory.")

current_chat_id = st.session_state.current_chat_id
current_chat = st.session_state.chats[current_chat_id]
vector_store = st.session_state.vector_store
convo_chain = build_conversational_chain(vector_store, current_chat["memory"])

if current_chat["history"]:
    st.markdown("<div class='chat-meta'>Recent messages</div>", unsafe_allow_html=True)
for q, a in current_chat["history"][-20:]:
    st.markdown(f"<div class='user-q'>You: {escape(q)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-a'>Bot: {escape(a)}</div>", unsafe_allow_html=True)

nonce = st.session_state.nonces.get(current_chat_id, 0)
q_key = f"q_input_{current_chat_id}_{nonce}"
question = st.text_input("Enter your question:", key=q_key)

if st.button("Submit", type="primary", key=f"submit_{current_chat_id}"):
    if question.strip():
        with st.spinner("Thinking..."):
            result = convo_chain({"question": question.strip()})
            answer = result.get("answer", "").strip()
        current_chat["history"].append((question.strip(), answer))
        if len(current_chat["history"]) == 1 and question.strip():
            t = question.strip()
            current_chat["title"] = (t[:30] + "…") if len(t) > 30 else t
        st.session_state.nonces[current_chat_id] = nonce + 1
        st.rerun()
    else:
        st.warning("Please enter a question to get an answer.")
