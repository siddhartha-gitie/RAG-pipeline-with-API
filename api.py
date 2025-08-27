import os
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA




data_path = Path("data")
files = list(data_path.rglob("*.txt"))


docs = []
for file in files:
    loader = TextLoader(file.as_posix())
    docs.extend(loader.load())



splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)



retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="llama3")  # Make sure you already pulled this model locally
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


app = FastAPI(title="RAG Pipeline API", version="1.0")


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



class QueryRequest(BaseModel):
    question: str



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = rag_chain.invoke(request.question)
        # If result is a dict with "result" key, extract string, otherwise use as-is
        if isinstance(result, dict) and "result" in result:
            answer_text = result["result"]
        else:
            answer_text = result
        return {"question": request.question, "answer": answer_text}
    except Exception as e:
        return {"error": str(e)}
