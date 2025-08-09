# quick_test.py - Quick test of core components

import os
from dotenv import load_dotenv
load_dotenv()

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from src.core.agent.graph import build_agent_graph, APP_STATE
        from src.core.agent.agent_state import AgentState
        from src.core.data_processing.vectors import build_scalable_retriever_system
        from src.core.agent.memory import ConversationVectorMemory
        from sqlalchemy import create_engine
        print("[PASS] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_database():
    """Test database connection."""
    print("Testing database...")
    try:
        from sqlalchemy import create_engine, inspect
        DB_URL = os.getenv('DB_URL', 'sqlite:///DATA/plm_updated.db')
        engine = create_engine(DB_URL)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"[PASS] Database connected. Tables: {tables}")
        return len(tables) > 0
    except Exception as e:
        print(f"[FAIL] Database test failed: {e}")
        return False

def test_openai_key():
    """Test OpenAI API key."""
    print("Testing OpenAI API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and len(api_key) > 10:
        print("[PASS] OpenAI API key found")
        return True
    else:
        print("[FAIL] OpenAI API key not found or invalid")
        return False

def test_retriever_system():
    """Test retriever system initialization."""
    print("Testing retriever system...")
    try:
        from sqlalchemy import create_engine
        from src.core.data_processing.vectors import build_scalable_retriever_system
        import json
        
        DB_URL = os.getenv('DB_URL', 'sqlite:///DATA/plm_updated.db')
        engine = create_engine(DB_URL)
        
        # Load schema description
        with open('src/config/schema_description.json', 'r') as f:
            schema_data = json.load(f)
        
        retrievers = build_scalable_retriever_system(engine, schema_data)
        print(f"[PASS] Retriever system built successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Retriever system test failed: {e}")
        return False

def test_agent_graph():
    """Test agent graph building."""
    print("Testing agent graph...")
    try:
        from src.core.agent.graph import build_agent_graph
        graph = build_agent_graph()
        print("[PASS] Agent graph built successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Agent graph test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("QUICK SYSTEM TEST")
    print("="*50)
    
    tests = [
        test_imports,
        test_openai_key,
        test_database,
        test_retriever_system,
        test_agent_graph
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"[FAIL] Test failed with exception: {e}")
            print()
    
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! System is ready.")
        return True
    else:
        print("WARNING: Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)