# 📚 Retrieval-Augmented Generation (RAG) Pipeline with API

This project implements a **RAG pipeline** using [LangChain](https://www.langchain.com/), FAISS for vector search, and local/remote LLMs (OpenAI or Ollama).  
It also provides an **API layer (FastAPI)** to interact with the RAG system programmatically.

---

## 🚀 Features
- Load documents from a `data/` folder.
- Chunk and embed text using `RecursiveCharacterTextSplitter`.
- Store and search vectors using **FAISS**.
- Query the knowledge base with **RAG pipeline**.
- Run a **FastAPI server** to expose the pipeline as an API.
- Supports both **OpenAI** and **Ollama (local LLMs)**.

---

## 🛠️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/rag-pipeline-api.git
cd rag-pipeline-api
```

### 2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Prepare Data
Add .txt files inside the data/ directory.

###5. Run the RAG Script
python rag.py

### 6. Start the API
uvicorn api:app --reload


⚡ Example API Usage

Query the RAG system:

curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is LangChain used for?"}'


🔧 Requirements

Python 3.9+

Ollama
 (if using local models)

OpenAI API Key (if using OpenAI embeddings/LLMs)

📂 Project Structure
rag-pipeline-api/
│── data/           # store your .txt documents
│── rag.py          # main RAG pipeline script
│── api.py          # FastAPI server
│── requirements.txt
│── README.md
