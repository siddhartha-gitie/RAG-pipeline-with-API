import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

data_path = r"C:\Users\siddh\OneDrive\Desktop\PERSISTENT\TASK_3\RAG_PIPELINE\data"
files = list(Path(data_path).rglob("*.txt"))

docs = []
for file in files:
    loader = TextLoader(file.as_posix())
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(split_docs, embeddings)

retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3")

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What is LangChain used for?"
result = rag_chain.invoke(query)
print("\nAnswer:\n", result)
