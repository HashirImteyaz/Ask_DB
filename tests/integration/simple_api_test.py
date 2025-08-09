# simple_api_test.py - Simple API test

import requests
import json
import time
import subprocess
import threading
import os

def start_server():
    """Start the server."""
    subprocess.run(["python", "main_app.py"], cwd=os.getcwd())

def test_api():
    """Test the API with a few simple queries."""
    
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(20):
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=2)
            if response.status_code == 200:
                print("[PASS] Server is running")
                break
        except:
            time.sleep(1)
    else:
        print("[FAIL] Server failed to start")
        return
    
    # Upload schema
    print("Uploading schema...")
    try:
        with open('src/config/schema_description.json', 'rb') as f:
            files = {'context_file': f}
            response = requests.post("http://127.0.0.1:8000/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            print("[PASS] Schema uploaded")
        else:
            print(f"[FAIL] Schema upload failed: {response.status_code}")
            return
    except Exception as e:
        print(f"[FAIL] Schema upload error: {e}")
        return
    
    # Wait a moment for initialization
    time.sleep(3)
    
    # Test queries
    test_questions = [
        "How many specifications are there?",
        "What are the different business units?",
        "List all ingredient names"
    ]
    
    print(f"\nTesting {len(test_questions)} queries...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] Testing: {question}")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"query": question, "history": []},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                sql = data.get('sql', 'No SQL generated')
                answer = data.get('answer', 'No answer provided')
                
                print(f"    SQL: {sql[:100]}...")
                print(f"    Answer: {answer[:100]}...")
                print(f"    [PASS] Query successful")
            else:
                print(f"    [FAIL] HTTP {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"    [FAIL] Error: {e}")
    
    print(f"\nTest completed!")

if __name__ == "__main__":
    test_api()