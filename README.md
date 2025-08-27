# 📚 Retrieval-Augmented Generation (RAG) Pipeline with API and Frontend UI

This project implements a **RAG pipeline** using [LangChain](https://www.langchain.com/), FAISS for vector search, and local/remote LLMs (OpenAI or Ollama).  
It provides a **FastAPI backend API** alongside a simple **web frontend** for interactive querying.

---

## 🚀 Features

- Load documents from a `data/` folder (.txt files).
- Chunk and embed text using `RecursiveCharacterTextSplitter` and **Hugging Face embeddings**.
- Store and search vectors efficiently with **FAISS**.
- Query knowledge base with a **RAG pipeline** using local or remote LLMs.
- Serve a **FastAPI server** exposing both API and frontend UI endpoints.
- Supports **OpenAI** embeddings/LLMs or **local Ollama models** (e.g., LLaMA 3).
- Interactive web UI for live question answering via browser.

---

## 🛠️ Setup Instructions

### 1. Clone Repository

git clone https://github.com/your-username/rag-pipeline-api.git
cd rag-pipeline-api

text

### 2. Create Virtual Environment

python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows

text

### 3. Install Dependencies

pip install -r requirements.txt

text

### 4. Prepare Data

Add your `.txt` documents to the `data/` directory for indexing.

### 5. Run RAG Script (Optional)

python rag.py

text

### 6. Start the API and Frontend Server

uvicorn api:app --reload

text

Open http://127.0.0.1:8000/ in your browser to access the interactive frontend.

---

## ⚡ Example API Usage

Query the RAG API:

curl -X POST "http://127.0.0.1:8000/query"
-H "Content-Type: application/json"
-d '{"question": "What is LangChain used for?"}'

text

---

## 📂 Project Structure

rag-pipeline-api/
│── data/ # Text files serving as knowledge base
│── templates/ # HTML frontend template files
│── static/ # (Optional) Static files for frontend (CSS/JS)
│── rag.py # Main RAG pipeline script (indexing + querying)
│── api.py # FastAPI server code with frontend integration
│── requirements.txt
│── README.md

text

---

## 🔧 Requirements

- Python 3.9 or higher
- Ollama (for local LLMs) - https://ollama.com/
- OpenAI API key (optional, if using OpenAI embeddings/LLMs)
- Internet access for initial model pulls (Hugging Face, Ollama models)

---

## 📖 About the Project

This project demonstrates a full Retrieval-Augmented Generation pipeline combining state-of-the-art embedding models and language models with an easy-to-use API and a browser frontend. This allows querying your own documents intelligently and privately without relying solely on cloud APIs.

---

## 🚀 Next Steps & Contributions

- Extend frontend UI with better styling and user experience.
- Add support for streaming responses from LLMs.
- Integrate additional vector stores or databases.
- Welcome contributions! Please submit issues or pull requests.

---
