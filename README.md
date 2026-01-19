# EnterpriseGPT

EnterpriseGPT is an **internal AI workspace** designed for an organization.  
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

![EnterpriseGPT](https://github.com/user-attachments/assets/f9e512e1-949a-4969-b343-2fe6e10a6ddf)

1. **Types of Users** 
    - There are 2 types of Users. **Admin** and **Employee**.
    - **Admin** can upload resumes and policies.
    - **Employee** can ask questions to the chatbot.
      
2. **Application UI** 
    - The UI is built using **AntiGravity IDE** from **Google** using Vibe Coding.
    - It has two interfaces. **HR Portal** and **Employee Portal**.
    - Frontend stack used includes **React JS** and **Vanilla CSS**.
      
3. **Cloud Infrastructure**
    - The **backend** is built using **FastAPI** and is deployed on a **Debian VM** and is accessed via **HTTP**.
    - The **Vector database** uses **ChromaDB** for **vector search** and **metadata storage** deployed on separate **Debian VM**.
    - The **structured data storage** uses **BigQuery**.
    - The **file storage** uses **Google Cloud Storage**.
    - The **Orchestration** uses **LangChain**.
    - The **LLM** uses **OpenAI**.
      
## Motivation
For the last few years, I have been part of a great learning curve wherein I have upskilled myself to move into a Machine Learning and Cloud Computing. This project was practice project for all the learnings I have had. This is first of the many more to come.

## Libraries/frameworks used

<b>Built with</b>
- [Python](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [BigQuery](https://cloud.google.com/bigquery)
- [Google Cloud Storage](https://cloud.google.com/storage)
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [React JS](https://reactjs.org/)
- [Vanilla CSS](https://developer.mozilla.org/en-US/docs/Web/CSS)

## Cloning Repository

```bash
    # clone this repo:
    git clone https://github.com/adityasolanki205/EnterpriseGPT.git
```

## Initial Setup

Below are the steps to setup the enviroment and run the codes:
 
1. **Setup**: First we will have to setup free google cloud account which can be done [here](https://cloud.google.com/free).

2. **IDE Setup**: Download Antigravity from [here](https://antigravity.google/). This is only required if any changes are required.

3. **Prototype (Optional)**: A Quick prototype can be created [here](https://aistudio.google.com/apps)

## Cloud infrastructure setup:

1. Goto **Google Compute Engine** and use below configuraiton to create 2 Virtual machines:

  - **VM 1**: For **Backend**
    - **Name**: enterprisegpt-backend
    - **Region**: asia-south2
    - **Machine type**: e2-medium
    - **Operating system**: Debian 11
    - **Data Protection**: No Backups
    - **Disk size**: 20 GB
    - **Firewall**: Allow HTTP and HTTPS traffic
    - **Security**: Allow full access to all Cloud APIs
    - **Automation**:
      ```bash
        sudo apt update && sudo apt upgrade -y
        sudo apt install -y python3-pip python3-venv nodejs npm nginx git
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt install -y nodejs
      ```
  
  - **VM 2**: For **ChromaDB**
    - **Name**: enterprisegpt-chromadb
    - **Region**: asia-south2
    - **Machine type**: e2-medium
    - **Operating system**: Debian 11
    - **Data Protection**: No Backups
    - **Disk size**: 20 GB
    - **Firewall**: Allow HTTP and HTTPS traffic
    - **Security**: Allow full access to all Cloud APIs
    - **Automation**:
      ```bash
        sudo apt update && sudo apt upgrade -y
        sudo apt install -y python3-pip python3-venv nodejs npm nginx git
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt install -y nodejs
      ```

2. Goto **Google Cloud Storage** and use below configuraiton to create 1 bucket:
    - **Name**: enterprisegpt-bucket
    - **Region**: asia-south2
    - **Storage class**: Standard
    - **Access**: Public access to allUsers

3. Goto **Google bigquery** and use below configuraiton to create 1 dataset:
    - **Name**: enterprisegpt-dataset
    - **Region**: asia-south2

4. Goto **Google bigquery** and use below configuraiton to create 1 Table:
    - **Name**: employee_data
    - **Region**: asia-south2
    - **Schema**:
      ```json
          [
            {
                "name": "id",
                "type": "STRING",
                "mode": "NULLABLE"
            },
            {
                "name": "name",
                "type": "STRING",
                "mode": "REQUIRED"
            },
            {
                "name": "department",
                "type": "STRING",
                "mode": "NULLABLE"
            },
            {
                "name": "is_on_bench",
                "type": "BOOLEAN",
                "mode": "REQUIRED"
            },
            {
                "name": "bench_start_date",
                "type": "DATE",
                "mode": "NULLABLE"
            },
            {
                "name": "project_id",
                "type": "STRING",
                "mode": "NULLABLE"
            },
            {
                "name": "allocation_pct",
                "type": "INTEGER",
                "mode": "NULLABLE"
            },
            {
                "name": "last_updated_at",
                "type": "DATE",
                "mode": "NULLABLE"
            }
          ]
      ```
    - **Sample Data**:
      ```sql
      INSERT INTO `<project_name>.enterprisegpt.employee_data`
          (
              id,
              name,
              department,
              is_on_bench,
              bench_start_date,
              project_id,
              allocation_pct,
              last_updated_at
          )
          VALUES
          -- Employee 1: On Bench
          (
              'E001',
              'Aditya Solanki',
              'Engineering',
              TRUE,
              DATE '2025-01-10',
              NULL,
              0,
              CURRENT_DATE()
          ),

          -- Employee 2: Fully Allocated
          (
              'E002',
              'Pratibha Singh',
              'Data',
              FALSE,
              NULL,
              'PRJ-101',
              100,
              CURRENT_DATE()
          ),

          -- Employee 3: Partially Allocated
          (
              'E003',
              'Rahul Verma',
              'Engineering',
              FALSE,
              NULL,
              'PRJ-102',
              50,
              CURRENT_DATE()
          ),

          -- Employee 4: Recently on Bench
          (
              'E004',
              'Vikram Bhatt',
              'QA',
              TRUE,
              DATE '2025-02-01',
              NULL,
              0,
              CURRENT_DATE()
          );
      ```


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

