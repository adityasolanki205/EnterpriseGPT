import os
import subprocess
import sys
import argparse

def run_chroma_server(host: str = "0.0.0.0", port: int = 8001, persist_directory: str = "chroma_db"):
    """
    Starts the ChromaDB server using the 'chroma run' command.
    This script should be deployed and run on the separate Virtual Machine.
    """
    print(f"--- ChromaDB Server Deployment Script ---")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Storage Path: {os.path.abspath(persist_directory)}")
    print(f"-----------------------------------------")
    
    # Ensure the directory exists
    if not os.path.exists(persist_directory):
        print(f"Creating storage directory: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
    
    # Command to run chroma server
    # We use sys.executable -m chromadb.cli.command run ... if possible, 
    # but 'chroma' command is the standard entry point.
    # To be safer with python environments, we'll try to find the executable or run via module if known.
    # However, 'chroma' is the documented CLI.
    
    command = [
        "chroma", "run",
        "--path", persist_directory,
        "--host", host,
        "--port", str(port)
    ]
    
    print(f"Executing command: {' '.join(command)}")
    print("Use Ctrl+C to stop the server.")
    
    try:
        # On Windows, shell=True might be needed if chroma is a batch file, 
        # but usually subprocess handles executables in path well.
        # If 'chroma' is not in PATH, this will fail.
        # Fallback to python -m chromadb if possible? chromadb doesn't support 'python -m chromadb' directly universally.
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nERROR: 'chroma' command not found.")
        print("Please ensure 'chromadb' is installed in the current environment:")
        print("  pip install chromadb")
        print("If it is installed, ensure the Scripts directory is in your PATH.")
    except KeyboardInterrupt:
        print("\nStopping ChromaDB server...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChromaDB Server on a Remote VM")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--path", default="chroma_db", help="Path to local storage directory on the VM")
    
    args = parser.parse_args()
    
    run_chroma_server(args.host, args.port, args.path)
