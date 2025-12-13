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
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

@app.get("/")
def read_root():
    return {"message": "Enterprise GPT API is running"}

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    processed_count = 0
    try:
        all_chunks = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            chunk = load_and_process_document(file_path)
            all_chunks.extend(chunk)
            processed_count += 1
        
        if all_chunks:
            vectorstore.add_documents(all_chunks) # Use global vectorstore
            
        return {
            "status": "success", 
            "message": f"Successfully processed {processed_count} documents."
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Global memory storage (Simple version for demo)
history_store = {} 

@app.post("/chat")
async def chat_endpoint(message: str = Form(...), portal: str = Form(...)):
    try:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # --- Session handling ---
        session_id = "default"  # replace with real session/user id later
        if session_id not in history_store:
            history_store[session_id] = InMemoryChatMessageHistory()

        chat_history = history_store[session_id]

        # --- System role based on portal ---
        if portal == "hr":
            system_prompt = (
                "You are an expert HR Assistant. "
                "Use the provided context to answer questions about candidates and internal documents.\n\n"
                "Context:\n{context}"
            )
        else:
            system_prompt = (
                "You are a helpful Employee Support Assistant. "
                "Use the provided context to answer questions about company policies, leave, and benefits.\n\n"
                "Context:\n{context}"
            )

        # --- LLM ---
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0
        )

        # --- Prompt (LCEL style) ---
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])

        # --- RAG Chain ---
        chain = (
            {
                "context": retriever | format_docs,  # <--- Pipe retrieved docs into formatter
                "question": RunnablePassthrough(),
                "chat_history": lambda _: chat_history.messages,
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        # --- Invoke chain ---
        answer = chain.invoke(message)

        # --- Update chat history explicitly ---
        chat_history.add_user_message(message)
        chat_history.add_ai_message(answer)

        return {"response": answer}

    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
