import os
import shutil
import re
from typing import List, Dict, Any, Optional
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
from google.cloud import storage
from google.cloud import bigquery
import chromadb # Added for HttpClient

"""
EnterpriseGPT Backend Application
=================================

This module serves as the backend for the EnterpriseGPT application.
It uses FastAPI to provide API endpoints for document processing,
chat functionality, and integration with:
- OpenAI (LLM & Embeddings)
- Google Cloud Storage (File storage)
- Google BigQuery (Employee data analytics)
- ChromaDB (Vector database for RAG)
- LangChain (Orchestration context)

Key features:
- Document upload, classification (Resume vs Policy), and indexing.
- Context-aware chat using RAG (Retrieval-Augmented Generation).
- SQL generation for querying structured employee data in BigQuery.
"""

# Load .env
load_dotenv() 

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
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Remote ChromaDB Configuration
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost") # Replace with VM IP
CHROMA_SERVER_PORT = int(os.getenv("CHROMA_SERVER_PORT", 8001))
PROJECT_ID = os.getenv("PROJECT_ID","solar-dialect-264808")
DATASET = os.getenv("DATASET","enterprisegpt")
TABLE = os.getenv("TABLE","employee_data")

FULL_TABLE_NAME = f"`{PROJECT_ID}.{DATASET}.{TABLE}`"

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set.")

# --- Helper Functions ---

def get_vectorstore():
    """
    Initializes and returns the Chroma vector store.
    
    Connects to a remote ChromaDB server running on a separate VM.
    Uses OpenAIEmbeddings for vectorization.
    
    Returns:
        Chroma: An instance of the LangChain Chroma vector store.
    
    Raises:
        Exception: If connection to the ChromaDB server fails.
    """
    embeddings = OpenAIEmbeddings()
    # Connect to the remote ChromaDB server running on the separate VM
    # The server should be started using chroma_server_deploy.py
    try:
        client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)
        return Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name="langchain" # Default collection name
        )
    except Exception as e:
        print(f"Failed to connect to ChromaDB at {CHROMA_SERVER_HOST}:{CHROMA_SERVER_PORT}. Error: {e}")
        raise e


def generate_sql(question: str) -> str:
    """
    Generates a SQL query based on a natural language question using an LLM.
    
    Args:
        question (str): The user's question about employee data.
        
    Returns:
        str: The generated SQL query.
    """
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0
    )

    chain = SQL_PROMPT | llm

    sql = chain.invoke({
        "question": question,
        "FULL_TABLE_NAME": FULL_TABLE_NAME
    }).content.strip()

    return sql

def run_bigquery(sql: str):
    """
    Executes a SQL query against Google BigQuery.
    
    Args:
        sql (str): The SQL query string.
        
    Returns:
        List[dict]: A list of rows returned by the query, where each row is a dictionary.
    """
    client = bigquery.Client(project=PROJECT_ID)
    job = client.query(sql)
    return [dict(row) for row in job.result()]

def query_bench_employees(question: str):
    """
    Retrieves bench employees data from BigQuery based on a natural language question.
    
    Args:
        question (str): The user's question about employee data.
        
    Returns:
        List[dict]: A list of rows returned by the query, where each row is a dictionary.
    """
    sql = generate_sql(question)
    print("Generated SQL:\n", sql)  
    return run_bigquery(sql)

def fetch_resume_links_for_bench(bench_names: set[str]):
    """
    Retrieves resume GCS links from VectorDB for a given set of employee names.
    
    Args:
        bench_names (set[str]): A set of names of employees on the bench.
        
    Returns:
        Dict[str, str]: A dictionary mapping employee names to their resume GCS links.
    """
    resume_links = {}

    for name in bench_names:
        # Query the vector store for resume documents matching the employee name
        results = vectorstore.get(
            where={
                "$and":[
                    {"doc_type": "resume"},
                    {"employee_name": name}
                ]
            },
            include=["metadatas"]
        )

        # Extract the GCS link from metadata
        for meta in results.get("metadatas", []):
            if meta and meta.get("gcs_link"):
                resume_links[name] = meta["gcs_link"]
                break

    return resume_links

def langchain_bench_with_resumes(prompt: str):
    """
    Handles queries about bench employees by combining Structured Data (BigQuery) 
    and Unstructured Data (Resume Links from VectorDB).
    
    Args:
        prompt (str): The user's input/question.
        
    Returns:
        List[dict]: A list combining BigQuery results with resume links.
    """
    bench_rows = query_bench_employees(prompt)

    bench_names = {row["name"] for row in bench_rows}
    # Enhance the structured data with unstructured resume links
    resume_links = fetch_resume_links_for_bench(bench_names)

    final = []
    for row in bench_rows:
        final.append({
            "employee_name": row["name"],
            "bench_since": row["bench_start_date"],
            "resume_link": resume_links.get(row["name"], "Not available")
        })

    return final

def is_bench_question(question: str) -> bool:
    """
    Checks if a question is related to bench/unallocated employees.
    
    Args:
        question (str): The user's question.
        
    Returns:
        bool: True if the question relates to bench status, False otherwise.
    """
    keywords = [
        "bench",
        "on bench",
        "available employees",
        "free employees",
        "unallocated",
        "not assigned to project"
    ]
    q = question.lower()
    return any(k in q for k in keywords)

def upload_to_gcs(source_file_path: str, destination_blob_name: str):
    """
    Uploads a file to Google Cloud Storage (GCS).
    
    Args:
        source_file_path (str): The local path to the file.
        destination_blob_name (str): The destination path/name in the bucket.
        
    Returns:
        str | None: The public URL of the uploaded file if successful, else None.
    """
    if not GCS_BUCKET_NAME:
        print("Skipping GCS upload: GCS_BUCKET_NAME not set.")
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        print(f"File {source_file_path} uploaded to {destination_blob_name}.")
        return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{destination_blob_name}"
    except Exception as e:
        print(f"Failed to upload to GCS: {e}")
        return None 

def load_and_process_document(file_path: str):
    """
    Loads a document and splits it into chunks.
    
    Args:
        file_path (str): The absolute path to the document file.
        
    Returns:
        List[Document]: A list of document chunks.
    """
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
    # Split text into chunks to respect token limits and improve retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
    splits = text_splitter.split_documents(docs)
    return splits

nlp = spacy.load("en_core_web_sm")

def extract_features(text: str) -> dict:
    """
    Extracts heuristic features from text to help classify it as 'Resume' or 'Policy'.

    Features include presence of emails, phone numbers, bullets, specific keywords 
    (sections like 'Experience' vs 'Governance'), and entity counts (Person vs Org).

    Args:
        text (str): The content of the document.

    Returns:
        dict: A dictionary of feature counts/booleans.
    """
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
            "policy", "scope", "objective", "purpose",
            "guidelines", "compliance", "procedure",
            "applicability", "responsibility",
            "governance", "effective date", "revision",
            "approval", "authority", "definitions"
            ]),
        "modal_verbs": sum(v in text_l for v in ["must", "shall", "required","may not", "prohibited"]),
        "governance_terms": sum(t in text_l for t in ["act", "law", "regulation", "iso", "government", "ministry"]),
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
    """
    Classifies a document as 'resume' or 'policy' based on feature scores.
    
    Args:
        text (str): The document text.
        
    Returns:
        str: 'resume' or 'policy'.
    """
    f = extract_features(text)

    # Heuristic scoring for Resumes
    resume_score = (
        f["has_email"] * 2 +
        f["has_phone"] * 2 +
        f["years"] +
        f["resume_sections"] * 2 +
        f["bullets"] +
        f["person_entities"]
    )
    print(resume_score)

    # Heuristic scoring for Policies
    policy_score = (
        f["policy_sections"] * 2 +
        f["modal_verbs"] * 2 +
        f["org_entities"] +
        f["governance_terms"] * 3 +
        (3 if f["bullets"] == 0 else 0)
    )
    print(policy_score)
    return "resume" if resume_score >= policy_score else "policy"

RESUME_STOPWORDS = {
    "resume", "cv", "profile", "document", "updated", "latest"
}

def extract_employee_name_from_filename(filename: str) -> str | None:
    """
    Extracts and formats an employee's name from a filename.
    Expected format: first_last_something.ext
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        str | None: The formatted name (e.g., "John Doe") or None if extraction fails.
    """
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

def get_all_employee_names() -> set[str]:
    """
    Retrieves all unique employee names stored in the VectorDB metadata.
    
    Returns:
        set[str]: A set of employee names.
    """
    results = vectorstore.get(include=["metadatas"])
    names = set()

    for meta in results["metadatas"]:
        if meta and "employee_name" in meta:
            names.add(meta["employee_name"])

    return names

def is_resume_question(question: str) -> bool:
    """
    Determines if a question is asking about a specific employee/resume.
    It checks if any known employee name appears in the question.
    
    Args:
        question (str): The user's question.
        
    Returns:
        bool: True if it contains an employee name, False otherwise.
    """
    question_lower = question.lower()
    employee_names = get_all_employee_names()
    print(employee_names)
    for name in employee_names:
        if name.lower() in question_lower:
            return True
    return False

# --- API Endpoints ---

# Global Vectorstore Initialization
# Note: This will attempt to connect on startup. Ensure the remote server is running.
vectorstore = get_vectorstore()

@app.get("/")
def read_root():
    return {"message": "Enterprise GPT API (Remote DB) is running"}

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint to process uploaded documents (PDF, DOCX, TXT).
    
    Workflow:
    1. Saves the file locally.
    2. Loads and splits the document into chunks.
    3. Classifies the document as 'resume' or 'policy'.
    4. Extracts metadata (e.g., employee name).
    5. Uploads the file to Google Cloud Storage (GCS).
    6. Adds document chunks with metadata to the VectorDB.
    """
    processed_count = 0
    try:
        all_chunks = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            docs = load_and_process_document(file_path)
            print (docs)
            if not docs:
                continue

            # Classify based on the first few chunks
            sample_text = " ".join(d.page_content for d in docs[:3])
            print(sample_text)
            doc_type = classify_document(sample_text)
            
            # Extract name if it's a resume (or try from filename)
            # employee_name = extract_employee_name(sample_text) if doc_type == "resume" else None
            employee_name = extract_employee_name_from_filename(file.filename)
            print(doc_type)
            print(employee_name)  
            
            # Upload original file to GCS
            gcs_link = upload_to_gcs(file_path, f"{doc_type}/{file.filename}") 
            print(gcs_link)
            
            # Update metadata for each chunk
            for d in docs:
                d.metadata.update({
                    "doc_type": doc_type,
                    "employee_name": employee_name,
                    "source_file": file.filename,
                    "gcs_link": gcs_link
                })
            all_chunks.extend(docs)
            processed_count += 1
        
        if all_chunks:
            vectorstore.add_documents(all_chunks) # Use global vectorstore (remote)
            
        return {
            "status": "success", 
            "message": f"Successfully processed {processed_count} documents."
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
# Global memory storage (Simple version for demo)

history_store = {}

RESUME_PROMPT = """
You are an expert HR Assistant. 
Use the provided context to answer questions about candidates.\n\n
If the information is not found in the context, explicitly state that you don't know based on the documents.\n\n
Context:\n{context}
IMPORTANT:
You must return ONLY valid JSON in the following format.
Do NOT add markdown, explanations, bullet points, or any text outside the JSON object.

{{
"employee_name": "Full name of the employee",
"summary": "2–3 line professional summary based strictly on the resume",
"skills": ["skill1", "skill2"],
"total_experience": "X years",
"companies_worked_in": ["Company A", "Company B"],
"resume_link": "The source link provided in the context"
}}
"""

POLICY_PROMPT = """
You are an HR Policy Assistant.

Answer the user's question strictly using the provided context.
• If the context lists types (e.g., Public Holidays, Restricted Holidays), enumerate them.
• If numeric counts are explicitly mentioned, extract and report them clearly.
• If counts vary yearly or are not defined, explicitly state that the document does not specify fixed counts.
• If numbers appear near holiday descriptions, prioritize extracting them.
• If the context provides a source link, include it in the response.
• Do NOT infer, assume, or hallucinate numbers.

Context:
{context}
"""

SQL_PROMPT = ChatPromptTemplate.from_template("""
You are an HR analytics assistant.

Write a BigQuery SELECT query ONLY.

Rules:
- ONLY SELECT queries
- Use this table exactly: {FULL_TABLE_NAME}
- Use column names exactly as they exist
- Column name for status of bench is "is_on_bench" and stores boolean values
- Do NOT explain anything
- Do NOT use markdown

User question:
{question}

Return ONLY SQL.
""")

@app.post("/chat")
async def chat_endpoint(message: str = Form(...), portal: str = Form(...)):
    """
    Chat endpoint handling user queries.
    
    Logic:
    1. Checks if the query is about "Bench" employees -> Triggers SQL/BigQuery flow.
    2. Checks if the query is about a specific "Resume" -> Filters VectorDB for resumes.
    3. Default -> Filters VectorDB for "Policy" documents.
    4. Retrieves relevant context and generates a response using GPT-4.
    5. Updates session chat history.
    """
    try:
        def format_docs(docs):
            """Formats retrieved documents for the LLM context."""
            formatted_chunks = []
            for doc in docs:
                content = doc.page_content
                link = doc.metadata.get("gcs_link", "N/A")
                formatted_chunks.append(f"Content: {content}\nSource Link: {link}")
            formatted = "\n\n".join(formatted_chunks)
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
        
        # --- Intent Detection ---
        is_resume = is_resume_question(message)
        
        # 1. Handle Bench Questions (Structured Data)
        if is_bench_question(message):
            data = langchain_bench_with_resumes(message)

            table = (
            "| Name | On Bench Since | Resume Link |\n"
            "|------|---------------|-------------|\n"
            )

            for r in data:
                table += (
                    f"| {r['employee_name']} | {r['bench_since']} | "
                    f"[Download]({r['resume_link']}) |\n"
                    )

            return {"response": table}
            
        # 2. Define filter based on question type (Unstructured Data)
        if is_resume:
             filter_dict = {"doc_type": "resume"}
        else:
             filter_dict = {"doc_type": "policy"}

        # Create a new retriever with the filter
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 20, "filter": filter_dict}
        )

        llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0
        )
        print (is_resume)
        
        # 3. Select Prompt & Parser based on context
        if is_resume:
            system_prompt = RESUME_PROMPT
            output_parser = JsonOutputParser()
        else:
            system_prompt = POLICY_PROMPT
            output_parser = StrOutputParser()
            
        prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("placeholder", "{chat_history}"),
                    ("human", "{question}")
                ])

        # 4. Construct and execute the RAG chain
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
        
        # 5. Format response
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
                        + "\n- **Download Resume**:\n" 
                        + f"  - [Click here to download]({answer.get('resume_link', '#')})\n"
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