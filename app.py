import os
import io
import time
import uuid
import requests
import pdfplumber
import streamlit as st

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


# -----------------------------
# Google API Key Setup
# -----------------------------
GOOGLE_API_KEY = "AIzaSyDxd5WI0-AcioR9KqjESVElivPzdk-QLo8"
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# -----------------------------
# Styling
# -----------------------------
st.set_page_config(page_title="📚 PDF Question Answering Bot", layout="wide")

st.markdown(
    """
    <style>
        .reportview-container { background-color: #f4f4f9; color: #333; }
        .sidebar .sidebar-content { background-color: #f0f0f5; }
        h1 { color: #004c8c; }
        .stButton>button { background-color: #0066ff; color: white; }
        .user-q { color: #0066ff; font-weight: bold; margin-top: 10px; }
        .bot-a { color: #333; margin-left: 10px; margin-bottom: 15px; }
        .chat-meta { font-size: 12px; color: #666; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# PDF helpers
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
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,  # no sources
        verbose=False,
    )
    return chain


# -----------------------------
# App Boot: load PDF & vector store once per session
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
# Multi-chat state
# -----------------------------
def _new_chat(title: str | None = None):
    """Create a new chat with its own memory & empty history."""
    chat_id = str(uuid.uuid4())
    chat_title = title or f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {
        "title": chat_title,
        "created_at": int(time.time()),
        "history": [],  # list of (q, a)
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        ),
    }
    st.session_state.current_chat_id = chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    _new_chat("Chat 1")  # initialize first chat


# -----------------------------
# Sidebar: chat list + new chat
# -----------------------------
st.sidebar.header("Chats")

# Build choices for selectbox
chat_ids = list(st.session_state.chats.keys())
choices = [st.session_state.chats[cid]["title"] for cid in chat_ids]
current_idx = chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else 0

selected_title = st.sidebar.selectbox("Select a chat", choices, index=current_idx, key="chat_selector")

# Map title back to id
for cid in chat_ids:
    if st.session_state.chats[cid]["title"] == selected_title:
        if cid != st.session_state.current_chat_id:
            st.session_state.current_chat_id = cid
        break

if st.sidebar.button("➕ New Chat", use_container_width=True):
    _new_chat()
    st.rerun()


# -----------------------------
# UI: conversation for selected chat
# -----------------------------
st.title("📚 PDF Question Answering Bot")
st.subheader(st.session_state.chats[st.session_state.current_chat_id]["title"])
st.caption("Ask questions about the financial policy PDF. Each chat has its own memory.")

current_chat = st.session_state.chats[st.session_state.current_chat_id]
vector_store = st.session_state.vector_store

# Build chain for this chat (uses its own memory)
convo_chain = build_conversational_chain(vector_store, current_chat["memory"])

# Conversation history at the top
if current_chat["history"]:
    st.markdown("<div class='chat-meta'>Recent messages</div>", unsafe_allow_html=True)
for i, (q, a) in enumerate(current_chat["history"][-20:], 1):
    st.markdown(f"<div class='user-q'>You: {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-a'>Bot: {a}</div>", unsafe_allow_html=True)

# Input at the bottom
question = st.text_input("Enter your question:", key=f"q_input_{st.session_state.current_chat_id}")

if st.button("Submit", type="primary", key=f"submit_{st.session_state.current_chat_id}"):
    if question.strip():
        with st.spinner("Thinking..."):
            result = convo_chain({"question": question.strip()})
            answer = result.get("answer", "").strip()
        # Save to this chat only
        current_chat["history"].append((question.strip(), answer))
        # Optional: set a better title from first user question
        if len(current_chat["history"]) == 1 and question.strip():
            current_chat["title"] = (question.strip()[:30] + "…") if len(question.strip()) > 30 else question.strip()
        st.rerun()
    else:
        st.warning("Please enter a question to get an answer.")
