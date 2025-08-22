# AnnBot
Ask questions about a PDF—like a financial policy—and get precise, context-grounded answers.  
Built with **Streamlit**, **LangChain**, **FAISS**, **Sentence-Transformers**, and **Google Gemini**.
- **Live app:** https://askannbot.streamlit.app/  
- **Repository:** https://github.com/TasfiaTahsinAnnita/AnnBot


---

## Features

- **PDF Q&A (RAG)** – Splits a PDF into chunks, embeds them with `all-MiniLM-L6-v2`, retrieves with **FAISS**, and answers using **Gemini**.
- **Per-chat memory** – Each chat keeps its own conversation history; no cross-leak.
- **Multiple chats** – Create/select chats in the sidebar; input clears after submit.
- **Clean answers** – No noisy citations/sources in the output.
- **Quota-friendly** – Uses **Gemini 1.5 Flash** and automatically falls back to a lighter model if rate/usage limits are hit.
- **Branding** – Sidebar logo from a raw image URL.

---

## Tech Stack

- **UI:** Streamlit  
- **LLM:** Google Gemini (`gemini-1.5-flash-latest`, fallback: `gemini-1.5-flash-8b`)  
- **RAG:** LangChain `ConversationalRetrievalChain`  
- **Embeddings:** Sentence-Transformers `all-MiniLM-L6-v2`  
- **Vector Store:** FAISS  
- **PDF Parsing:** pdfplumber

---

## Project Structure

```
AnnBot/
├─ app.py
├─ requirements.txt
└─ (assets like logo / PDFs live in the repo)
```

---

## Quick Start (Local)

> Requires **Python 3.10+**

```bash
# 1) Clone
git clone https://github.com/TasfiaTahsinAnnita/AnnBot
cd AnnBot

# 2) (Recommended) Create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
streamlit run app.py
```

By default, the app:
- Downloads the target PDF from the repo’s raw URL  
- Shows the logo in the sidebar  
- Uses Gemini 1.5 Flash, falling back to a lighter model if quota is exhausted

---

## Configuration

### PDF Source

Update the source PDF in `app.py` (use a raw URL so it downloads correctly):

```python
PDF_URL = "https://github.com/TasfiaTahsinAnnita/AnnBot/raw/main/For%20Task%20-%20Policy%20file.pdf"
```

### Sidebar Logo

Use a raw image URL (GitHub raw works well):

```python
LOGO_URL = "https://raw.githubusercontent.com/TasfiaTahsinAnnita/AnnBot/main/Annbotlogo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)
```

Tip: Convert a GitHub page URL  
`https://github.com/.../blob/main/logo.png` → `https://raw.githubusercontent.com/.../main/logo.png`

### Model / Retrieval Settings

Tune inside the chain builder:

```python
model_name="gemini-1.5-flash-latest",  # or "gemini-1.5-flash-8b"
k=4,                                   # top-k retrieved chunks
max_output_tokens=512                  # answer length
```

### API Key

The app uses Google’s Generative AI (Gemini). In `app.py` it’s configured like:

```python
GOOGLE_API_KEY = "YOUR_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

Production recommendation: store the key outside code, e.g.

**Environment variable**
```bash
export GOOGLE_API_KEY="your-key"
streamlit run app.py
```

**Streamlit secrets (`.streamlit/secrets.toml`)**
```toml
GOOGLE_API_KEY = "your-key"
```

```python
import streamlit as st
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
```

---

## How It Works (RAG Flow)

1. **Ingest** — Download PDF & extract text via `pdfplumber`.  
2. **Chunk** — Split into ~1000-character chunks with overlap.  
3. **Embed** — Create embeddings using `all-MiniLM-L6-v2`.  
4. **Index** — Store vectors in **FAISS**.  
5. **Retrieve** — For each question, fetch top-k similar chunks.  
6. **Generate** — Feed context + question to **Gemini** for the final answer.  
7. **Memory** — Keep per-chat history for coherent follow-ups.

---

## Requirements

`requirements.txt`:

```
streamlit
pdfplumber
requests
google-generativeai
langchain
langchain-community
langchain-google-genai
faiss-cpu
sentence-transformers
PyPDF2
```

---

## Troubleshooting

- **ResourceExhausted / rate limit**  
  Built-in fallback to a lighter Gemini model. If it persists, try shorter queries or slower cadence.

- **FAISS import error**  
  Ensure `faiss-cpu` is installed (it’s included in `requirements.txt`).

- **No text extracted**  
  Scanned PDFs may need OCR (e.g., integrate Tesseract) as `pdfplumber` extracts text only, not images.

- **Logo not visible**  
  Use a raw image URL (not a GitHub page). Example:  
  `https://raw.githubusercontent.com/<user>/<repo>/<branch>/<path>/<file>.png`

---


## Acknowledgements

- Streamlit  
- LangChain  
- FAISS  
- Sentence-Transformers  
- pdfplumber  
- Google Gemini

---

## License

Choose a license (e.g., MIT) and add it to the repo.
