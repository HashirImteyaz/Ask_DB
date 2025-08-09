# manual_test.py - Manual test without server startup

import requests
import json
import time

def test_running_server():
    """Test a server that's already running."""
    base_url = "http://127.0.0.1:8000"
    
    print("Testing server availability...")
    
    # Test server health
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("[PASS] Server is accessible")
        else:
            print(f"[FAIL] Server returned {response.status_code}")
            return
    except Exception as e:
        print(f"[FAIL] Cannot connect to server: {e}")
        print("Please start the server first with: python main_app.py")
        return
    
    # Upload schema if needed
    print("Uploading schema description...")
    try:
        with open('src/config/schema_description.json', 'rb') as f:
            files = {'context_file': f}
            response = requests.post(f"{base_url}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            print("[PASS] Schema uploaded successfully")
        else:
            print(f"[WARN] Schema upload returned {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"[WARN] Schema upload failed: {e}")
        print("Continuing with tests...")
    
    # Wait for initialization
    time.sleep(2)
    
    # Test queries from the evaluation set
    test_questions = [
        "How many total recipes are there in the system?",
        "What are all the ingredient names and codes?",
        "List all recipe names with their status",
        "What is the country and BU code for each recipe manufactured?"
    ]
    
    print(f"\nTesting {len(test_questions)} NLQ queries...")
    
    successful_tests = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}] {question}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"query": question, "history": []},
                timeout=45,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                sql = data.get('sql', 'No SQL generated')
                answer = data.get('answer', 'No answer provided')
                
                # Evaluate response
                sql_generated = bool(sql and sql != 'No SQL generated')
                answer_provided = bool(answer and answer != 'No answer provided')
                
                print(f"  Response Time: {response_time:.0f}ms")
                print(f"  SQL Generated: {'Yes' if sql_generated else 'No'}")
                if sql_generated:
                    print(f"  SQL: {sql[:80]}...")
                print(f"  Answer Provided: {'Yes' if answer_provided else 'No'}")
                if answer_provided:
                    print(f"  Answer: {answer[:100]}...")
                
                if sql_generated and answer_provided:
                    print(f"  [PASS] Test successful")
                    successful_tests += 1
                else:
                    print(f"  [FAIL] Missing SQL or answer")
            else:
                print(f"  [FAIL] HTTP {response.status_code}")
                print(f"  Error: {response.text[:200]}")
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            print(f"  [FAIL] Exception after {response_time:.0f}ms: {e}")
    
    # Summary
    success_rate = successful_tests / len(test_questions)
    print(f"\n" + "="*60)
    print(f"TEST SUMMARY")
    print(f"="*60)
    print(f"Successful Tests: {successful_tests}/{len(test_questions)}")
    print(f"Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.75:
        print(f"[EXCELLENT] System is working very well!")
    elif success_rate >= 0.5:
        print(f"[GOOD] System is working adequately")
    else:
        print(f"[NEEDS IMPROVEMENT] System has significant issues")
    
    return success_rate >= 0.5

if __name__ == "__main__":
    print("="*60)
    print("MANUAL NLQ SYSTEM TEST")
    print("="*60)
    print("This test requires the server to be running.")
    print("To start the server: python main_app.py")
    print("="*60)
    
    success = test_running_server()
    exit(0 if success else 1)