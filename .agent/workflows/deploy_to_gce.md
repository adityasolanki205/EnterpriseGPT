---
description: Deploy the EnterpriseGPT application (FastAPI + React) on Google Cloud Compute Engine
---

This workflow guides you through deploying the EnterpriseGPT application to a Google Cloud Compute Engine (VM instance) running Ubuntu.

## Prerequisites
1.  **Google Cloud Platform (GCP) Account**: You must have an active GCP account with billing enabled.
2.  **GCP Project**: Create or select an existing project.
3.  **Local Environment**: Ensure you have `gcloud` CLI installed or use the Google Cloud Console.

## Step 1: Create a VM Instance

1.  Go to **Compute Engine** > **VM instances** in the Google Cloud Console.
2.  Click **Create Instance**.
3.  **Name**: `enterprisegpt-vm` (or your preferred name).
4.  **Region/Zone**: Select one close to you (e.g., `us-central1-a`).
5.  **Machine type**: `e2-medium` (2 vCPU, 4GB memory) is sufficient for this app.
6.  **Boot disk**:
    *   OS: **Ubuntu**
    *   Version: **Ubuntu 22.04 LTS** (x86/64)
    *   Size: 10-20 GB Balanced persistent disk.
7.  **Firewall**:
    *   Check **Allow HTTP traffic**.
    *   Check **Allow HTTPS traffic**.
8.  Click **Create**.

## Step 2: Configure Firewall Rules (Optional but Recommended)
By default, the backend runs on port 8000. For production, we will use Nginx to reverse proxy port 80 to 8000, so opening port 8000 externally is not strictly necessary if we follow this guide. However, if you want to test the backend directly:
1.  Go to **VPC network** > **Firewall**.
2.  Create a firewall rule named `allow-8000`.
3.  Targets: `All instances in the network`.
4.  Source filter: `IPv4 ranges`, Source ranges: `0.0.0.0/0`.
5.  Protocols and ports: `tcp:8000`.
6.  Click **Create**.

## Step 3: Connect to the VM
1.  Click the **SSH** button next to your instance in the Cloud Console to open a browser-based terminal.

## Step 4: Install System Dependencies
Run the following commands in the SSH terminal to update the system and install necessary tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nodejs npm nginx git
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

## Step 5: Clone the Repository
Clone your project repository. Since your code is local, you might want to push it to GitHub/GitLab first, or use `scp` to copy files.
*Assuming you pushed to GitHub:*

```bash
git clone https://github.com/adityasolanki205/EnterpriseGPT.git
cd EnterpriseGPT
```

*Alternatively, upload your local files manually.*

## Step 6: Setup Backend (FastAPI)

1.  **Navigate to backend directory**:
    ```bash
    cd backend
    ```

2.  **Create Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

4.  **Setup Environment Variables**:
    Create a `.env` file with your API keys.
    ```bash
    nano .env
    ```
    Paste your variables:
    ```
    OPENAI_API_KEY=sk-proj-...
    ```
    Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

5.  **Test Backend**:
    Not to be done
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    Visit `http://<VM_EXTERNAL_IP>:8000` in your browser. You should see `{"message": "Enterprise GPT API is running"}`.
    Press `Ctrl+C` to stop.

6.  **Setup as a System Service (Keep it running)**:
    Create a systemd file:
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
    Start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl start enterprisegpt-backend
    sudo systemctl enable enterprisegpt-backend
    sudo systemctl status enterprisegpt-backend
    ```

## Step 7: Setup Frontend (React)

1.  **Update API URL**: 
    ## Not required
    Before building, you must point the frontend to your VM's IP or Domain.
    Edit `frontend/src/App.jsx`:
    ```javascript
    const API_URL = "http://<VM_EXTERNAL_IP_OR_DOMAIN>/api"; // We will use /api in Nginx
    // OR if exposing port 8000 directly: "http://<VM_EXTERNAL_IP>:8000"
    ```
    *Tip: Best practice is to use an environment variable like `VITE_API_URL` and access it via `import.meta.env.VITE_API_URL`.*

2.  **Build the Frontend**:
    ```bash
    cd ../frontend
    npm install
    npm run build
    ```
    This creates a `dist` folder.

3.  **Deploy Static Files to Nginx**:
    Copy build files to the web root:
    ```bash
    sudo mkdir -p /var/www/enterprisegpt
    sudo cp -r dist/* /var/www/enterprisegpt/
    ```

## Step 8: Configure Nginx (Reverse Proxy)

Configure Nginx to serve the frontend and proxy API requests to the backend.

1.  Create Nginx config:
    ```bash
    sudo vi /etc/nginx/sites-available/enterprisegpt
    ```

2.  Add configuration:
    
    ```
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


    Enable the site:
    ```bash
    sudo ln -s /etc/nginx/sites-available/enterprisegpt /etc/nginx/sites-enabled/
    sudo rm /etc/nginx/sites-enabled/default
    sudo nginx -t
    sudo systemctl restart nginx
    ```

## Step 9: Verify Deployment
1.  Open `http://<VM_EXTERNAL_IP>` in your browser.
2.  You should see the React app.
3.  Upload a document or chat to test backend connectivity.


#Restart Backend Service

sudo systemctl stop enterprisegpt-backend
sudo systemctl status enterprisegpt-backend
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl start enterprisegpt-backend
sudo systemctl status enterprisegpt-backend


# curl commands to test:
curl http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/chat -F "message=hello" -F "portal=employee"
curl http://localhost/api/chat -F "message=hello" -F "portal=employee"


# restart frontend: 
sudo nginx -t
sudo systemctl reload nginx



Chroma db setup:
sudo vi /etc/systemd/system/chroma.service

[Unit]
Description=Chroma Vector Database Service
After=network.target

[Service]
Type=simple
User=aditya_solanki205
WorkingDirectory=/home/aditya_solanki205/EnterpriseGPT/backend/chroma

ExecStart=/home/aditya_solanki205/EnterpriseGPT/backend/venv/bin/chroma run \
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


sudo systemctl daemon-reload
sudo systemctl enable chroma
sudo systemctl start chroma



sudo mkdir -p /opt/chroma
sudo mkdir -p /var/lib/chroma
sudo chown -R aditya_solanki205:aditya_solanki205 /opt/chroma /var/lib/chroma

cd /opt/chroma
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install chromadb

sudo vi /etc/systemd/system/chroma.service

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

sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart chroma

sudo -u aditya_solanki205 ls /opt/chroma
sudo -u aditya_solanki205 ls /var/lib/chroma

ss -lntp | grep 8001
journalctl -u chroma -n 100 --no-pager

curl -v http://<VM_internal_IP>:8001

sudo chown -R aditya_solanki205:aditya_solanki205 /var/lib/chroma
sudo chmod 755 /var/lib/chroma


