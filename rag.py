import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Path to your local data directory
data_path = r"C:\Users\siddh\OneDrive\Desktop\PERSISTENT\TASK_3\RAG_PIPELINE\data"
files = list(Path(data_path).rglob("*.txt"))

# Load documents
docs = []
for file in files:
    loader = TextLoader(file.as_posix())
    docs.extend(loader.load())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Use HuggingFace embeddings (no API key required)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Use Ollama with LLaMA3 (make sure Ollama is installed and running)
llm = Ollama(model="llama3")

# Build RAG chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a query
query = "What is LangChain used for?"
result = rag_chain.run(query)
print("\nAnswer:\n", result)
