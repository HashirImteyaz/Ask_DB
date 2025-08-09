#!/usr/bin/env python3
"""
Test script to validate the conversation memory fix for follow-up questions.
Tests the specific scenario: "What are the names of recipes?" followed by 
"Which countries are involved in making these recipes?"
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_conversation_scenario():
    print("=" * 80)
    print("TESTING CONVERSATION MEMORY FIX")
    print("Scenario: Ask about recipes, then ask follow-up about 'these recipes'")
    print("=" * 80)
    
    # Step 1: Start a new session and ask about recipes
    print("\n🔹 Step 1: Asking 'What are the names of recipes?'")
    
    response1 = requests.post(f"{API_BASE}/chat", json={
        "query": "What are the names of recipes?",
        "session_id": "test_conversation_fix"
    })
    
    if response1.status_code != 200:
        print(f"❌ Error in step 1: {response1.status_code} - {response1.text}")
        return
    
    result1 = response1.json()
    print(f"✅ Response 1: {result1['answer'][:200]}...")
    
    # Wait a moment to ensure the conversation is stored
    time.sleep(1)
    
    # Step 2: Ask the follow-up question with "these recipes"
    print("\n🔹 Step 2: Asking 'Which countries are involved in making these recipes?'")
    
    response2 = requests.post(f"{API_BASE}/chat", json={
        "query": "Which countries are involved in making these recipes?",
        "session_id": "test_conversation_fix"
    })
    
    if response2.status_code != 200:
        print(f"❌ Error in step 2: {response2.status_code} - {response2.text}")
        return
    
    result2 = response2.json()
    print(f"✅ Response 2: {result2['answer'][:300]}...")
    
    # Analyze the results
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    
    if "clarification" in result2['answer'].lower() or ("which" in result2['answer'].lower() and "recipe" not in result2['answer'].lower()):
        print("❌ FAILED: System asked for clarification instead of resolving 'these recipes'")
        print(f"   Response: {result2['answer']}")
    else:
        print("✅ SUCCESS: System resolved 'these recipes' and provided relevant answer")
        print(f"   Response contained recipe context: {'recipe' in result2['answer'].lower()}")
    
    # Check if the response includes specific recipe names or countries
    has_countries = any(word in result2['answer'].lower() for word in ['country', 'countries', 'italy', 'france', 'spain', 'greece'])
    if has_countries:
        print("✅ Response includes country information")
    else:
        print("⚠️  Response doesn't clearly mention countries")
    
    print("\n" + "=" * 80)
    print("Test completed. Check the responses above to verify the fix.")
    return result1, result2

if __name__ == "__main__":
    try:
        test_conversation_scenario()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
