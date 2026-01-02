import os
import shutil
import re
from typing import List
from collections import Counter

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
import spacy

# Load .env
load_dotenv() 

#app = FastAPI()
app = FastAPI(
    title="Enterprise GPT",
    root_path="/api",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

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

nlp = spacy.load("en_core_web_sm")

def extract_features(text: str) -> dict:
    text_l = text.lower()

    features = {
        "has_email": bool(re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)),
        "has_phone": bool(re.search(r"\b\d{10}\b|\+\d{1,3}[\s-]?\d{7,10}", text)),
        "years": len(re.findall(r"\b(19|20)\d{2}\b", text)),
        "bullets": text.count("•") + text.count("- "),
        "resume_sections": sum(s in text_l for s in [
            "experience", "education", "skills", "projects", "certification"
        ]),
        "policy_sections": sum(s in text_l for s in [
            "policy", "scope", "guidelines", "compliance", "procedure"
        ]),
        "modal_verbs": sum(v in text_l for v in ["must", "shall", "required"])
    }

    doc = nlp(text)
    ent_counts = Counter(ent.label_ for ent in doc.ents)

    features.update({
        "person_entities": ent_counts.get("PERSON", 0),
        "org_entities": ent_counts.get("ORG", 0)
    })
    print(features)
    return features
def classify_document(text: str) -> str:
    f = extract_features(text)

    resume_score = (
        f["has_email"] * 2 +
        f["has_phone"] * 2 +
        f["years"] +
        f["resume_sections"] * 2 +
        f["bullets"] +
        f["person_entities"]
    )
    print(resume_score)

    policy_score = (
        f["policy_sections"] * 2 +
        f["modal_verbs"] * 2 +
        f["org_entities"] +
        (3 if f["bullets"] == 0 else 0)
    )
    print(policy_score)
    return "resume" if resume_score >= policy_score else "policy"

RESUME_STOPWORDS = {
    "resume", "cv", "profile", "document", "updated", "latest"
}

def extract_employee_name_from_filename(filename: str) -> str | None:
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")

    if len(parts) < 2:
        return None

    first_name = parts[0].strip()
    last_name = parts[1].strip()

    if not first_name or not last_name:
        return None
    print(first_name)
    print(last_name)    
    return f"{first_name.title()} {last_name.title()}"

# --- API Endpoints ---

# Global Vectorstore Initialization
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}
)

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
            
            docs = load_and_process_document(file_path)
            if not docs:
                continue

            sample_text = " ".join(d.page_content for d in docs[:3])
            print(sample_text)
            doc_type = classify_document(sample_text)
            #employee_name = extract_employee_name(sample_text) if doc_type == "resume" else None
            employee_name = extract_employee_name_from_filename(file.filename)
            print(doc_type)
            print(employee_name)   
            for d in docs:
                d.metadata.update({
                    "doc_type": doc_type,
                    "employee_name": employee_name,
                    "source_file": file.filename    
                })
            all_chunks.extend(docs)
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
def get_all_employee_names() -> set[str]:
    results = vectorstore.get(include=["metadatas"])
    names = set()

    for meta in results["metadatas"]:
        if meta and "employee_name" in meta:
            names.add(meta["employee_name"])

    return names

def is_resume_question(question: str) -> bool:
    question_lower = question.lower()
    employee_names = get_all_employee_names()
    print(employee_names)
    for name in employee_names:
        if name.lower() in question_lower:
            return True

    return False
# Global memory storage (Simple version for demo)

history_store = {}
RESUME_PROMPT = """
You are an expert HR Assistant. 
Use the provided context to answer questions about candidates.\n\n
If the information is not found in the context, explicitly state that you don't know based on the documents.\n\n
Context:\n{context}
IMPORTANT:
You must return ONLY valid JSON in the following format.
Do not add explanations, markdown, or extra text.

{{
"employee_name": "Full name of the employee",
"summary": "2–3 line professional summary based strictly on the resume",
"skills": ["skill1", "skill2"],
"total_experience": "X years",
"companies_worked_in": ["Company A", "Company B"]
}}
"""

GENERAL_PROMPT = """
You are a helpful assistant.
Answer the user's question using the provided context when relevant.
If the answer is not in the context, say so honestly.

Context:
{context}
"""

@app.post("/chat")
async def chat_endpoint(message: str = Form(...), portal: str = Form(...)):
    try:
        def format_docs(docs):
            formatted = "\n\n".join(doc.page_content for doc in docs)
            print("\n" + "=" * 50)
            print(f"DEBUG: Retrieved {len(docs)} chunks for context:")
            print("=" * 50)
            print(formatted)
            print("=" * 50 + "\n")
            return formatted

        # --- Session handling ---
        session_id = "default"  # replace with real session/user id later
        if session_id not in history_store:
            history_store[session_id] = InMemoryChatMessageHistory()

        chat_history = history_store[session_id]
        is_resume = is_resume_question(message)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        print (is_resume)
        if is_resume:
            system_prompt = RESUME_PROMPT   # must enforce JSON
            output_parser = JsonOutputParser()
        else:
            system_prompt = GENERAL_PROMPT    # NO JSON instruction
            output_parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("placeholder", "{chat_history}"),
                    ("human", "{question}")
                ])

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: chat_history.messages,
            }
            | prompt
            | llm
            | output_parser
        )
        answer = chain.invoke(message)
        if is_resume:
            response  = (
                        f"- **Name**:\n"
                        f"  - {answer.get('employee_name', 'Not found in documents')}\n"
                        f"- **Summary**:\n"
                        f"  - {answer.get('summary', 'Not found in documents')}\n"
                        "- **Skills**:\n"
                        + "\n".join(f"  - {s}" for s in answer.get("skills", []))
                        + "\n- **Total Experience**:\n"
                        f"  - {answer.get('total_experience', 'Not found in documents')}\n"
                        "- **Companies Worked In**:\n"
                        + "\n".join(f"  - {c}" for c in answer.get("companies_worked_in", []))
                        )
        else:
            response = answer
        print (response)


        # --- Update chat history explicitly ---
        chat_history.add_user_message(message)
        chat_history.add_ai_message(response)

        return {"response": response}

    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
