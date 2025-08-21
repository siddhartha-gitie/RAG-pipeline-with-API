import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# -------------------------
# Load Documents
# -------------------------
data_path = Path("data")
files = list(data_path.rglob("*.txt"))

docs = []
for file in files:
    loader = TextLoader(file.as_posix())
    docs.extend(loader.load())

# -------------------------
# Split Documents
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# -------------------------
# Vectorstore & Embeddings
# -------------------------
embeddings = OpenAIEmbeddings()  # Needs OPENAI_API_KEY in env
vectorstore = FAISS.from_documents(split_docs, embeddings)

# -------------------------
# Retriever + LLM + RAG Chain
# -------------------------
retriever = vectorstore.as_retriever()
llm = Ollama(model="tinyllama")
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI(title="RAG Pipeline API", version="1.0")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Welcome to the RAG Pipeline API!"}


@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = rag_chain.run(request.question)
        return {"question": request.question, "answer": result}
    except Exception as e:
        return {"error": str(e)}
