from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

import getpass
import os
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

router = APIRouter()

class QueryRequest(BaseModel):
    context: str
    query: str

def split_text_into_documents(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

@router.get("/test")
async def test():
    return {"message": "Working!"}


@router.post("/reg-agent")
async def reg_agent(request: QueryRequest):
    try:
        # Split context into chunks and store in vector store
        documents = split_text_into_documents(request.context)
        vector_store.add_documents(documents)

        # Perform similarity search with the query
        search_results = vector_store.similarity_search(request.query, k=1)

        if not search_results:
            return {"response": "No relevant context found."}

        # Generate response using LLM and the top result
        relevant_context = search_results[0].page_content
        response = llm.predict(f"Context: {relevant_context}\n\nQuery: {request.query}")
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
