# start_server.py - Start server and upload schema automatically

import os
import time
import requests
import json
import subprocess
import threading
from pathlib import Path

def upload_schema_description():
    """Upload schema description file to initialize the system."""
    schema_file = "src/config/schema_description.json"
    api_url = "http://127.0.0.1:8000/upload"
    
    if not Path(schema_file).exists():
        print(f"Warning: {schema_file} not found")
        return False
    
    print("Waiting for server to start...")
    # Wait for server to be ready
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=2)
            if response.status_code == 200:
                print("Server is ready!")
                break
        except:
            time.sleep(1)
    else:
        print("Server failed to start within 30 seconds")
        return False
    
    # Upload schema description
    try:
        with open(schema_file, 'rb') as f:
            files = {'context_file': f}
            response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code == 200:
            print("✅ Schema description uploaded successfully!")
            return True
        else:
            print(f"❌ Failed to upload schema: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error uploading schema: {e}")
        return False

def start_server():
    """Start the FastAPI server."""
    print("Starting NLQ server...")
    os.system("python main_app.py")

if __name__ == "__main__":
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Upload schema description
    time.sleep(2)  # Give server a moment to start
    schema_uploaded = upload_schema_description()
    
    if schema_uploaded:
        print("\n🎯 System is ready for testing!")
        print("📊 Run tests with: python test_nlq_system.py")
        print("🌐 Web interface: http://127.0.0.1:8000")
    else:
        print("\n⚠️ System started but schema upload failed")
        print("You may need to upload src/config/schema_description.json manually via the web interface")
    
    # Keep the script running
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down...")