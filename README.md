# EnterpriseGPT

EnterpriseGPT is an **internal AI workspace** designed for mid-to-large organizations.  
It combines **Retrieval-Augmented Generation (RAG)**, **structured enterprise data**, and **vector search** to answer HR, employee, and policy-related questions accurately and securely.

## Key Features

- Resume & Policy Search using RAG
- Hybrid AI Architecture (BigQuery + Chroma + LLM)
- Bench Employee Identification
- Resume Metadata & Secure Resume Links
- Policy Q&A from Uploaded Documents
- Separate HR and Employee Portals
- Cloud-native and GCP-ready deployment

## Architecture Overview

![Enterprise](https://github.com/user-attachments/assets/36462ce6-1ac7-4afe-b920-bf726e759f4b)

Frontend (Vue / React)
|
v
FastAPI Backend (EnterpriseGPT)
|
├── BigQuery (Structured Data)
│ └── Employee status (bench / active)
|
├── Chroma Vector DB (Separate VM)
│ ├── Resume embeddings
│ └── Policy embeddings
|
├── Google Cloud Storage (GCS)
│ └── Resume & policy documents
|
└── OpenAI / LLM
└── Reasoning & summarization

## 🧠 Data Ownership Model

| Data Type | Source of Truth |
|----------|----------------|
Employee bench/active status | BigQuery |
Employee basic details | BigQuery |
Resume text & embeddings | Chroma |
Resume download links | GCS |
Policy documents | Chroma |
Summaries & reasoning | LLM |

> ❗ Structured data is **never** derived from LLMs.

```
EnterpriseGPT/
├── backend/
│ ├── main.py
│ ├── chroma_client.py
│ ├── requirements.txt
│ ├── venv/
│ └── uploaded_docs/
│
├── frontend/
│ ├── src/
│ ├── build/
│ └── package.json
│
├── chroma/
│ ├── chroma.service
│ └── data/ # /var/lib/chroma on VM
│
└── README.md
```

## 🗄️ Vector Database (Chroma)

- Runs on a **separate Debian VM**
- Deployed as a **systemd service**
- Accessed via HTTP from backend

### Chroma Service Management

```bash
sudo systemctl status chroma
sudo systemctl restart chroma  
sudo systemctl enable chroma 
```

📄 Document Ingestion
Supported Formats

- PDF

- DOCX

- TXT

Automatic Classification

Documents are classified as:

- Resume

- Policy

Classification is based on:

- Structural patterns

- Keywords

- Entity recognition

- Resume-specific indicators (email, phone, experience)

Metadata Stored in Chroma
```json
{
  "doc_type": "resume",
  "employee_id": "E123",
  "employee_name": "Aditya Solanki",
  "resume_url": "https://storage.googleapis.com/..."
}
```
👥 Bench Employee Workflow

Execution Flow

1. Fetch bench employees from BigQuery

2. Fetch resume metadata from Chroma

3. Generate summaries using LLM

4. Return structured JSON to frontend

```json 
{
  "type": "bench_employee_list",
  "count": 2,
  "data": [
    {
      "employee_name": "Aditya Solanki",
      "department": "Engineering",
      "resume_url": "https://storage.googleapis.com/...",
      "resume_summary": "Backend engineer with experience in Python and GCP."
    }
  ]
}
```
🌐 API Endpoints

| Endpoint                      | Description                 |
| ----------------------------- | --------------------------- |
| `POST /api/chat`              | Main chat endpoint          |
| `POST /api/process-documents` | Upload resumes and policies |
| `GET /api/docs`               | Swagger API documentation   |
| `GET /api/health/chroma`      | Chroma connectivity check   |

🖥️ Frontend

- Built using Vue or React

- Uses /api/* routes via Nginx reverse proxy

- Supports:

    - HR Portal
    - Employee Portal
    - Resume upload
    - Bench employee table view

🚀 Deployment

Backend
```bash
source venv/bin/activate
sudo systemctl restart enterprisegpt-backend
```

Chroma
```bash
sudo systemctl restart chroma
```

Frontend
```bash
npm run build
sudo cp -r build/* /var/www/enterprisegpt/
sudo systemctl reload nginx
```
📌 Final Notes

EnterpriseGPT follows enterprise AI best practices:

- No hallucination for structured data
- Clear data ownership
- Scalable and auditable architecture
- Cloud-native design

