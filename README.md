# EnterpriseGPT

EnterpriseGPT is an **internal AI workspace** designed for an organization. It combines **Retrieval-Augmented Generation (RAG)**, **structured enterprise data**, and **vector search** to answer HR related questions on employees and policies accurately. You just have to upload the policies or resumes and ask questions, the AI will answer them. It is a **cloud-native** and **GCP-ready** application. 

## Key Features

- Resume & Policy Search using RAG
- Hybrid AI Architecture (BigQuery + Chroma + LLM)
- Bench Employee Identification
- Resume Metadata & Secure Resume Links
- Policy Q&A from Uploaded Documents
- Separate HR and Employee Portals
- Cloud-native and GCP-ready deployment

## Architecture Overview

![EnterpriseGPT](https://github.com/user-attachments/assets/7bb8a27f-9e1a-4cad-bceb-b2711e59327c)


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

2. **IDE Setup**: Download Antigravity from [here](https://antigravity.google/). This is only required if any changes are needed.

3. **Prototype (Optional)**: A Quick prototype can be created [here](https://aistudio.google.com/apps)

## Cloud infrastructure setup:

1. Goto **Google Compute Engine** and use below configuraiton to create 2 Virtual machines:

  - **VM 1**: For **Application frontend and backend**
    - **Name**: enterprisegpt-app
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

## Application setup:

1. Goto **enterprisegpt-backend** VM and click on SSH. Follow below steps to setup fastapi application:
    - Clone the repo using 
      ```bash
      git clone https://github.com/adityasolanki205/EnterpriseGPT.git
      ```

    - Goto the cloned repo
      ```bash
      cd EnterpriseGPT/backend
      ```

    - Create a virtual environment and activate it
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

    - Install the required dependencies
      ```bash
      pip install -r requirements.txt
      python -m spacy download en_core_web_sm
      ```

    - Set the environment variables at **/EnterpriseGPT/.env**
      ```bash
      OPENAI_API_KEY=<your_openai_api_key>
      GCS_BUCKET_NAME=enterprisegpt_bucket
      CHROMA_SERVER_HOST=<internal_chroma_VM_ip>
      CHROMA_SERVER_PORT=8001
      PROJECT_ID=<your_project_id>
      DATASET="enterprisegpt"
      TABLE="employee_data"
      ```

    - Setup as a System Service
      ```bash
      sudo vi /etc/systemd/system/enterprisegpt-backend.service
      ```

      Add configuration (adjust paths for your user name):
      ```ini
      [Unit]
      Description=Enterprise GPT Backend
      After=network.target

      [Service]
      User=aditya_solanki205
      WorkingDirectory=/home/aditya_solanki205/EnterpriseGPT/backend
      EnvironmentFile=/home/aditya_solanki205/EnterpriseGPT/.env
      ExecStart=/home/aditya_solanki205/EnterpriseGPT/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
      Restart=always

      [Install]
      WantedBy=multi-user.target
      ```

    - Start the service:
      ```bash
      sudo systemctl daemon-reexec
      sudo systemctl daemon-reload
      sudo systemctl start enterprisegpt-backend
      sudo systemctl enable enterprisegpt-backend
      sudo systemctl status enterprisegpt-backend
      ```
    
    - Note In case the backend server has to be restarted, use below commands:
      ```bash
      sudo systemctl stop enterprisegpt-backend
      sudo systemctl status enterprisegpt-backend
      sudo systemctl daemon-reexec
      sudo systemctl daemon-reload
      sudo systemctl start enterprisegpt-backend
      sudo systemctl status enterprisegpt-backend
      ```
2. Now lets setup **Frontend**. Follow below steps to setup React application:

    - Create the frontend
      ```bash
      cd ../frontend
      npm install
      npm run build
      ```

    - Deploy Static Files to Nginx 
      ```bash
      sudo mkdir -p /var/www/enterprisegpt
      sudo cp -r dist/* /var/www/enterprisegpt/
      ```

    - Setup as a System Service
      ```bash
      sudo vi /etc/nginx/sites-available/enterprisegpt
      ```
      
    -  Copy configuration in the opened file:
      ```ini
        server {
          listen 80;
          server_name _;
          
          client_max_body_size 50M;

          # ---------- Frontend ----------
          root /var/www/enterprisegpt;
          index index.html;

          location / {
            try_files $uri $uri/ /index.html;
          }

          location /api/ {
          proxy_pass http://127.0.0.1:8000;
          proxy_http_version 1.1;

          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;

          # Required for FastAPI / WebSockets
          proxy_set_header Upgrade $http_upgrade;
          proxy_set_header Connection "upgrade";

          proxy_read_timeout 300;
          proxy_connect_timeout 300;
          proxy_send_timeout 300;
      }
      }
      ```

    - Enable the site:
      ```bash
      sudo ln -s /etc/nginx/sites-available/enterprisegpt /etc/nginx/sites-enabled/
      sudo rm /etc/nginx/sites-enabled/default
      sudo nginx -t
      sudo systemctl restart nginx
      ```
    
    - Note In case the Frontend server has to be restarted, use below commands:
      ```bash
      sudo nginx -t
      sudo systemctl reload nginx
      ```
      
3. Goto **enterprisegpt-chromadb** VM and click on SSH. Follow below steps to setup chroma service: 
    - Create required directories and set ownership
      ```bash
      sudo mkdir -p /opt/chroma
      sudo mkdir -p /var/lib/chroma
      sudo chown -R aditya_solanki205:aditya_solanki205 /opt/chroma /var/lib/chroma        
      ```

    - Create a virtual environment and activate it
      ```bash
      cd /opt/chroma
      python3 -m venv venv
      source venv/bin/activate
      ```

    - Install the required dependencies
      ```bash
      pip install --upgrade pip
      pip install chromadb
      ```
    
    - Setup as a System Service
      ```bash
      sudo vi /etc/systemd/system/chroma.service
      ```

      Add configuration (adjust paths for your user name):
      ```ini
      [Unit]
      Description=Chroma Vector Database Service
      After=network.target

      [Service]
      Type=simple
      User=aditya_solanki205
      WorkingDirectory=/opt/chroma

      ExecStart=/opt/chroma/venv/bin/chroma run \
        --host 0.0.0.0 \
        --port 8001 \
        --path /var/lib/chroma

      Restart=always
      RestartSec=5
      Environment=PYTHONUNBUFFERED=1

      StandardOutput=journal
      StandardError=journal

      [Install]
      WantedBy=multi-user.target
      ```

    - Check the permissions:
      ```bash
      sudo -u aditya_solanki205 ls /opt/chroma
      sudo -u aditya_solanki205 ls /var/lib/chroma
      ```

    - Start the service:
      ```bash
      sudo systemctl daemon-reload
      sudo systemctl start chroma
      sudo systemctl enable chroma
      sudo systemctl status chroma
      ```

4. Goto **enterprisegpt-backend** VM and copy the public URL. Try opening the app using **http://External-IP_of_enterprisegpt-backend_VM>**. 

## Application Process Descriptions

### 1. Document Ingestion & Classification
The system supports uploading **PDF, DOCX, and TXT** documents via the HR Portal.
- **Processing**: Files are saved locally, then text is extracted and chunked.
- **Classification**: A hybrid heuristic approach (Regex + NLP) classifies documents:
  - **Resume**: Identified by email/phone patterns, "Experience" sections, and Person entities.
  - **Policy**: Identified by governance keywords ("Scope", "Compliance") and modal verbs.
- **Storage**:
  - Original files are uploaded to **Google Cloud Storage (GCS)**.
  - Text embeddings and metadata are stored in **ChromaDB**.

### 2. Chat & RAG Workflow
The Chatbot serves as a unified interface for three distinct retrieval types:

#### A. Structured Data Query (Bench Analytics)
*Triggered by questions like "Who is on bench?" or "unallocated employees"*
1. **NL-to-SQL**: LLM converts the user's question into a BigQuery SQL query.
2. **Execution**: The query runs against the `employee_data` table in BigQuery.
3. **Enrichment**: System fetches Resume Links from ChromaDB for the returned employees.
4. **Response**: A structured table is displayed with employee details and resume download links.

#### B. Resume Specific Chat
*Triggered by questions containing a specific employee's name*
1. **Filtering**: Usage of `doc_type: "resume"` filter in Vector Search.
2. **Extraction**: LLM extracts structured fields (Skills, Summary, Experience) from the resume chunks.
3. **Response**: A dedicated "Candidate Profile" card is rendered.

#### C. Policy Q&A
*Default workflow for general HR queries*
1. **Filtering**: Usage of `doc_type: "policy"` filter.
2. **Retrieval**: Semantic search retrieves relevant policy clauses.
3. **Response**: LLM generates a grounded answer citing the policy content.

### 3. Data Schema

**ChromaDB Metadata**:
```json
{
  "doc_type": "resume",      // or "policy"
  "employee_name": "Name",   // Extracted from filename
  "gcs_link": "https://...", // Link to file in GCS
  "source_file": "filename"
}
```

**BigQuery Schema (Employee Data)**:
- `id`, `name`, `department`, `is_on_bench` (BOOL), `project_id`, `allocation_pct`.


## API Endpoints

The backend is built with **FastAPI** and exposes the following RESTful endpoints.

### 1. Chat & Query Interface
- **Endpoint**: `POST /api/chat`
- **Description**: The central entry point for all user interactions. It intelligently routes the user's query to the appropriate engine (SQL generation vs. Vector RAG) based on intent classification.
- **Parameters**:
  - `message` (From Data): The natural language question from the user.
  - `portal` (From Data): Context identifier (e.g., 'HR' or 'Employee').
- **Returns**: A JSON object containing the answer text or a structured table for bench employees.

### 2. Document Processing
- **Endpoint**: `POST /api/process-documents`
- **Description**: Handles the ingestion of raw documents (PDF, DOCX, TXT).
- **Process**:
  1. **Upload**: Saves file to server.
  2. **Classify**: Determines if it's a Resume or Policy.
  3. **Cloud Storage**: Uploads original to Google Cloud Storage.
  4. **Vectorize**: Generates embeddings and stores them in ChromaDB with metadata.

### 3. System & Health
- **Endpoint**: `GET /api/docs`
  - **Description**: Interactive Swagger UI documentation for testing APIs.
- **Endpoint**: `GET /api/`
  - **Description**: Health check to verify backend is running.

📌 Final Notes

EnterpriseGPT follows enterprise AI best practices:

- No hallucination for structured data
- Clear data ownership
- Scalable and auditable architecture
- Cloud-native design
