# Enterprise GPT (React + FastAPI)

A premium RAG-based application for Enterprise Knowledge Management, featuring separate portals for HR (Data Ingestion) and Employees (Support Chat).

## 🚀 How to Run

You will need two terminal windows to run the Frontend and Backend simultaneously.

### 1. Start the Backend (FastAPI)
This handles the document processing (LangChain/ChromaDB) and chat logic.

```bash
cd enterprise_gpt_react
# Install dependencies (first time only)
pip install -r backend/requirements.txt

# Run the server
# Use this command if 'python' alias fails:
& "C:\Users\hp\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" backend/main.py
```
*Backend runs on `http://localhost:8000`*

### 2. Start the Frontend (React)
This launches the user interface.

```bash
cd enterprise_gpt_react/frontend
# Install dependencies (first time only)
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
Ensure you have a `.env` file in the root `LLM_Engineering_Learning` folder with your OpenAI API Key:
```
OPENAI_API_KEY=sk-...
```
