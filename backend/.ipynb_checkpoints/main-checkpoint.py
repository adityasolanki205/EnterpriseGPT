import os
import shutil
import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Basic LangChain Components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load .env
load_dotenv() 

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploaded_docs"
VECTOR_DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set.")

# --- Helper Functions ---

def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=VECTOR_DB_DIR, 
        embedding_function=embeddings
    )

def load_and_process_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return []
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# --- API Endpoints ---

# Global Vectorstore Initialization
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()

@app.get("/")
def read_root():
    return {"message": "Enterprise GPT API is running"}

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    processed_count = 0
    try:
        all_splits = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            splits = load_and_process_document(file_path)
            all_splits.extend(splits)
            processed_count += 1
        
        if all_splits:
            vectorstore.add_documents(all_splits) # Use global vectorstore
            
        return {
            "status": "success", 
            "message": f"Successfully processed {processed_count} documents."
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(message: str = Form(...), portal: str = Form(...)):
    try:
        # Global retriever is already initialized
        
        # Define System Prompt
        if portal == "hr":
            system_role = "You are an expert HR Assistant. Use the provided context to answer questions about candidates and internal documents."
        else:
            system_role = "You are a helpful Employee Support Assistant. Use the provided context to answer questions about company policies, leave, and benefits."
            
        template = """{system_role}
        
        Context:
        {context}
        
        Question: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        
        # Debugging: Check retrieved documents
        docs = vectorstore.similarity_search(message)
        print(f"DEBUG: Retrieved {len(docs)} documents for query: {message}")
        for i, doc in enumerate(docs):
            print(f"--- Doc {i+1} ---")
            print(doc.page_content[:200] + "...")
            
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough(), "system_role": lambda x: system_role}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(message)
        print(f"DEBUG: LLM Response: {response}")
        
        return {"response": response}
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
