#!/usr/bin/env python3
"""
Test script to validate the conversation memory fix for follow-up questions.
Tests the specific scenario: "What are the names of recipes?" followed by 
"Which countries are involved in making these recipes?" with unique session ID
"""

import requests
import json
import time
import random

API_BASE = "http://localhost:8000"

def test_conversation_scenario():
    print("=" * 80)
    print("TESTING CONVERSATION MEMORY FIX - BYPASS CACHE")
    print("Scenario: Ask about recipes, then ask follow-up about 'these recipes'")
    print("=" * 80)
    
    # Use random session ID to bypass cache
    session_id = f"test_fix_{random.randint(1000, 9999)}"
    print(f"Using session ID: {session_id}")
    
    # Step 1: Start a new session and ask about recipes
    print("\n🔹 Step 1: Asking 'What are the names of recipes?'")
    
    response1 = requests.post(f"{API_BASE}/chat", json={
        "query": "What are the names of recipes?",
        "session_id": session_id
    })
    
    if response1.status_code != 200:
        print(f"❌ Error in step 1: {response1.status_code} - {response1.text}")
        return
    
    result1 = response1.json()
    print(f"✅ Response 1 received (length: {len(result1['answer'])} chars)")
    
    # Wait a moment to ensure the conversation is stored
    time.sleep(2)
    
    # Step 2: Ask the follow-up question with "these recipes"
    print("\n🔹 Step 2: Asking 'Which countries are involved in making these recipes?'")
    
    response2 = requests.post(f"{API_BASE}/chat", json={
        "query": "Which countries are involved in making these recipes?",
        "session_id": session_id  # Same session to maintain conversation context
    })
    
    if response2.status_code != 200:
        print(f"❌ Error in step 2: {response2.status_code} - {response2.text}")
        return
    
    result2 = response2.json()
    print(f"✅ Response 2: {result2['answer'][:200]}...")
    
    # Analyze the results
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    
    if "clarification" in result2['answer'].lower():
        print("❌ FAILED: System asked for clarification instead of resolving 'these recipes'")
        print(f"   Full response: {result2['answer']}")
    else:
        print("✅ SUCCESS: System resolved 'these recipes' and provided relevant answer")
        print(f"   Response contained recipe context: {'recipe' in result2['answer'].lower()}")
    
    # Check if the response includes specific countries or SQL results
    has_countries = any(word in result2['answer'].lower() for word in ['country', 'countries', 'italy', 'france', 'spain', 'greece'])
    has_sql_data = 'table' in result2['answer'].lower() or 'sql' in result2['answer'].lower()
    
    if has_countries or has_sql_data:
        print("✅ Response includes data results (countries or SQL data)")
    else:
        print("⚠️  Response doesn't contain expected data results")
    
    print("\n" + "=" * 80)
    print("Test completed. Check the responses above to verify the fix.")
    return result1, result2

if __name__ == "__main__":
    try:
        test_conversation_scenario()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
