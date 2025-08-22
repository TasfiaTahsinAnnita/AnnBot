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

# Google API Key Setup
GOOGLE_API_KEY = "AIzaSyDxd5WI0-AcioR9KqjESVElivPzdk-QLo8"
genai.configure(api_key=GOOGLE_API_KEY) 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

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
from sentence_transformers import SentenceTransformer
import numpy as np

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
st.title("PDF Question Answering Bot")

# GitHub PDF URL
pdf_url = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"  # Corrected to raw PDF URL

# Download the PDF
pdf_file = download_pdf_from_github(pdf_url)

st.write("PDF loaded from GitHub. You can now ask questions!")

# Extract text from the PDF
raw_text = get_pdf_text(pdf_file)

# Split text into chunks and create vector store
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)

# User input for questions
question = st.text_input("Ask a question:")

if question:
    st.write("Bot response:")
    response = user_input(question)
    st.write(response)
