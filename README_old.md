# Enterprise GPT (React + FastAPI)

A premium RAG-based application for Enterprise Knowledge Management, featuring separate portals for HR (Data Ingestion) and Employees (Support Chat).

## 🚀 How to Run

You will need three terminal windows (or a separate VM for ChromaDB) to run the Full Stack application.

### 1. Start ChromaDB Server (Remote/Local)
The backend uses a standalone ChromaDB server (tcp).

```bash
# Run the deployment script
python backend/chroma_server_deploy.py
```
*Runs on `http://localhost:8001` by default.*

### 2. Start the Backend (FastAPI)
This handles the document processing (LangChain/ChromaDB), GCS uploads, and BigQuery analytics.

```bash
cd enterprise_gpt_react
# Install dependencies
pip install -r backend/requirements.txt

# Run the server
python backend/main.py
```
*Backend runs on `http://localhost:8000`*

### 3. Start the Frontend (React)
This launches the user interface.

```bash
cd enterprise_gpt_react/frontend
# Install dependencies
npm install

# Run the dev server
npm run dev
```
*Frontend runs on `http://localhost:5173`*

## 🔑 Login Credentials (Demo)

*   **HR Admin Portal**:
    *   Username: `admin`
    *   Password: `admin`
*   **Employee Support**:
    *   Username: `user`
    *   Password: `user`

## 🛠️ Configuration
Create a `.env` file in the root directory.

**Required:**
```env
OPENAI_API_KEY=sk-...
```

**Google Cloud Config (BigQuery & Storage):**
```env
GCS_BUCKET_NAME=your_bucket_name
PROJECT_ID=your_gcp_project_id
DATASET=enterprisegpt
TABLE=employee_data
# Ensure GOOGLE_APPLICATION_CREDENTIALS is set or you are authenticated via gcloud
```

**Remote ChromaDB Config:**
```env
CHROMA_SERVER_HOST=localhost # or VM IP
CHROMA_SERVER_PORT=8001
```
