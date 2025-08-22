import os
import io
import requests
import pdfplumber
import streamlit as st

from typing import List
from hashlib import md5

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
st.set_page_config(page_title="PDF Question Answering Bot", layout="wide")

st.markdown(
    """
    <style>
        .reportview-container { background-color: #f4f4f9; color: #333; }
        .sidebar .sidebar-content { background-color: #f0f0f5; }
        h1 { color: #004c8c; }
        .stButton>button { background-color: #0066ff; color: white; }
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


def build_conversational_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
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
        return_source_documents=False,  # no sources returned
        verbose=False,
    )
    return chain


# -----------------------------
# UI
# -----------------------------
st.title("📚 PDF Question Answering Bot")

st.sidebar.header("About This App")
st.sidebar.write(
    "This bot answers your questions based on the content of a PDF loaded from GitHub. "
    "It uses vector search and remembers the conversation."
)

pdf_url = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"

with st.spinner('Processing the PDF...'):
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
    convo_chain = build_conversational_chain(vector_store)

st.success('PDF loaded successfully! You can now ask questions.')

col1, col2 = st.columns(2)

with col1:
    st.header("PDF Content (first page preview)")
    preview = page_docs[0].page_content if page_docs else ""
    st.text_area("Preview Text", value=preview[:2000], height=300)
    st.caption("Note: The bot searches the entire PDF, not just this preview.")

with col2:
    st.header("Conversation")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Show conversation memory at the top
    if st.session_state.history:
        for i, (q, a) in enumerate(st.session_state.history[-6:], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
        st.markdown("---")

    # Question input below memory
    question = st.text_input("Enter your question:")

    if st.button('Submit'):
        if question:
            with st.spinner("Thinking..."):
                result = convo_chain({"question": question})
                answer = result.get("answer", "").strip()

            st.write("**Bot response:**")
            st.write(answer if answer else "_Answer is not available in the text._")

            st.session_state.history.append((question, answer))
        else:
            st.write("Please enter a question to get an answer.")
