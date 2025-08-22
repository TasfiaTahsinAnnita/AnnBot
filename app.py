import os
import requests
import pdfplumber  # Updated for better PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Google API Key Setup
GOOGLE_API_KEY = "AIzaSyDxd5WI0-AcioR9KqjESVElivPzdk-QLo8"
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Custom CSS for styling
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
    </style>
    """, unsafe_allow_html=True
)

# Function to download the PDF from GitHub repository
def download_pdf_from_github(pdf_url):
    response = requests.get(pdf_url)
    file_name = pdf_url.split('/')[-1]
    with open(file_name, 'wb') as f:
        f.write(response.content)
    return file_name

# Function to extract text from PDFs using pdfplumber
def get_pdf_text(pdf_docs):
    text = ""
    if isinstance(pdf_docs, str):
        pdf_docs = [pdf_docs]  # Wrap single path in a list

    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:  # Use pdfplumber for better text extraction
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get sentence embeddings
def get_sentence_embeddings(text_chunks):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(text_chunks)
    return embeddings

# Function to create a vector store with embeddings
def get_vector_store(text_chunks):
    embeddings = get_sentence_embeddings(text_chunks)
    text_embedding_pairs = list(zip(text_chunks, embeddings))
    vector_store = FAISS.from_embeddings(text_embedding_pairs, SentenceTransformer("all-MiniLM-L6-v2").encode)
    vector_store.save_local("faiss_index")

# Function for user input handling
def user_input(user_question):
    embeddings = get_sentence_embeddings([user_question])  # Get embedding for the question

    # Load the existing vector store
    vector_store = FAISS.load_local("faiss_index", SentenceTransformer("all-MiniLM-L6-v2").encode, allow_dangerous_deserialization=True)

    docs = vector_store.similarity_search_by_vector(embeddings[0])

    chain = get_conversional_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response['output_text']

# Function to get the conversational chain
def get_conversional_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the text", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Streamlit Interface
st.set_page_config(page_title="PDF Question Answering Bot", layout="wide")

# Title for the app
st.title("📚 PDF Question Answering Bot")

# Sidebar with an introduction
st.sidebar.header("About This App")
st.sidebar.write(
    "This bot answers your questions based on the content of a PDF document loaded directly from a GitHub repository. "
    "Just ask a question related to the document and get an answer!"
)

# GitHub PDF URL (Corrected to raw PDF URL)
pdf_url = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"

# Download the PDF
pdf_file = download_pdf_from_github(pdf_url)

# Show progress bar while the PDF is being processed
with st.spinner('Processing the PDF...'):
    raw_text = get_pdf_text(pdf_file)

# Split text into chunks and create a vector store
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)

st.success('PDF loaded successfully! You can now ask questions.')

# Layout with columns: Left column for PDF text and right column for the chatbot
col1, col2 = st.columns(2)

with col1:
    st.header("PDF Content")
    st.text_area("Raw Text from PDF", value=raw_text, height=300)

with col2:
    st.header("Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button('Submit'):
        if question:
            st.write("Bot response:")
            response = user_input(question)
            st.write(response)
        else:
            st.write("Please enter a question to get an answer.")

