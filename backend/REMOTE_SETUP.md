# Remote ChromaDB Deployment Guide

This guide explains how to deploy the Enterprise GPT application with a distributed architecture where the Chroma Vector Database runs on a separate Virtual Machine (VM).

## Architecture

*   **VM 1 (Application Server)**: Runs the FastAPI backend (`main_remote.py`). Connects to VM 2 for vector storage/retrieval.
*   **VM 2 (Database Server)**: Runs the ChromaDB server (`chroma_server_deploy.py`). Stores the actual embeddings and documents.

---

## 1. Setting up the Database Server (VM 2)

This machine will host the vector database.

### Prerequisites
*   Python 3.9+ installed.
*   Ensure **Port 8000** (or your chosen port) is open in the VM's firewall/security group to allow inbound traffic from VM 1.

### Installation
1.  Copy `chroma_server_deploy.py` to this VM.
2.  Install the required library:
    ```bash
    pip install chromadb
    ```

### Running the Server
Run the deployment script. By default, it runs on port 8000.

**Linux/Mac:**
```bash
python chroma_server_deploy.py --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
python chroma_server_deploy.py --host 0.0.0.0 --port 8000
```

*The server is now listening for requests. Note the IP address of this VM (e.g., `192.168.1.50`).*

---

## 2. Setting up the Application Server (VM 1)

This machine runs the main chat application.

### Prerequisites
*   The standard `requirements.txt` for the project must be installed.
*   `chromadb` client must be installed (included in requirements).

### Configuration
You need to tell the application where the remote ChromaDB server is located using environment variables.

**Linux/Mac:**
```bash
export CHROMA_SERVER_HOST="192.168.1.50" # Replace with VM 2's IP
export CHROMA_SERVER_PORT=8001
```

**Windows (PowerShell):**
```powershell
$env:CHROMA_SERVER_HOST="192.168.1.50" # Replace with VM 2's IP
$env:CHROMA_SERVER_PORT=8000
```

### Running the Application
Use the specialized `main_remote.py` file which is configured for remote connections.

```bash
uvicorn main_remote:app --host 0.0.0.0 --port 8001 --reload
```

---

## Troubleshooting

1.  **Connection Refused**:
    *   Ensure VM 2's firewall allows traffic on port 8000.
    *   Ensure `chroma_server_deploy.py` is running with `--host 0.0.0.0` (not localhost).
    
2.  **Version Mismatch**:
    *   Ensure both VMs are running compatible versions of `chromadb`. It is best to stick to the same version (e.g., `pip install chromadb==0.4.x`) on both.

3.  **Latency**:
    *   Since the DB is remote, network latency might slightly affect retrieval speed compared to a local DB. Ensure both VMs are in the same region/network for best performance.
