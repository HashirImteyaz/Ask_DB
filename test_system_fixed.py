#!/usr/bin/env python3
"""
NLQ PLM System - Test Script
Test the complete system functionality including session logging and token tracking
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_session_logging():
    """Test session logging functionality"""
    print("=" * 60)
    print("TESTING SESSION LOGGING SYSTEM")
    print("=" * 60)
    
    try:
        from src.core.agent.session_logger import SessionLogger, get_session_logger
        
        # Test session logger creation
        logger = SessionLogger()
        print(f"SUCCESS: Session logger created - ID: {logger.session_id}")
        print(f"   Log file: {logger.log_file_path}")
        
        # Test token counting
        test_text = "This is a test message for token counting functionality."
        tokens = logger.count_tokens(test_text)
        print(f"SUCCESS: Token counting: '{test_text[:30]}...' = {tokens} tokens")
        
        # Test basic logging
        logger.log_query_session(
            user_query="List all recipes with chicken ingredients",
            sql_query="SELECT DISTINCT s_ing.SpecDescription FROM Specifications s_cuc JOIN RecipeExplosion re ON s_cuc.SpecCode = re.CUCSpecCode JOIN Specifications s_ing ON re.INGSpecCode = s_ing.SpecCode WHERE UPPER(s_cuc.SpecDescription) LIKE UPPER('%chicken%') AND s_cuc.SpecGroupCode = 'CUC' LIMIT 100;",
            agent_response="I found 15 recipes that contain chicken ingredients. Here are the key findings: Top Chicken Recipes include Chicken Curry Sauce, Chicken Soup Base, and Grilled Chicken Marinade.",
            llm_calls=[
                {
                    "type": "clarification",
                    "model": "gpt-4o-mini",
                    "prompt": "Analyze user query and determine routing...",
                    "response": '{"is_greeting": false, "is_safe": true, "needs_clarification": false, "wants_graph": false}',
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "type": "sql_generation", 
                    "model": "gpt-4o-mini",
                    "prompt": "Generate SQL for: List all recipes with chicken ingredients",
                    "response": "SELECT DISTINCT s_ing.SpecDescription FROM Specifications...",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            context_used="Previous conversation about recipe searches",
            metadata={
                "graph_generated": False,
                "processing_time_seconds": 2.45,
                "context_retrieval_used": True
            }
        )
        print("SUCCESS: Test query logged successfully")
        
        # Test second query with graph
        time.sleep(1)  # Small delay to show different timestamps
        logger.log_query_session(
            user_query="Create a bar chart of top 10 ingredients",
            sql_query="SELECT DISTINCT s_ing.SpecDescription, COUNT(*) as ingredient_count FROM Specifications s_ing WHERE s_ing.SpecGroupCode = 'ING' GROUP BY s_ing.SpecDescription ORDER BY ingredient_count DESC LIMIT 10;",
            agent_response="I've created a bar chart showing the top 10 most frequently used ingredients. [CHART] Bar chart generated successfully!",
            llm_calls=[
                {
                    "type": "sql_generation",
                    "model": "gpt-4o-mini", 
                    "prompt": "Generate SQL for chart data: top 10 ingredients",
                    "response": "SELECT DISTINCT s_ing.SpecDescription, COUNT(*) as ingredient_count...",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            metadata={
                "graph_generated": True,
                "processing_time_seconds": 3.12,
                "chart_type": "bar"
            }
        )
        print("SUCCESS: Chart query logged successfully")
        
        # Get session stats
        stats = logger.get_session_stats()
        print(f"SUCCESS: Session stats: {stats['total_queries']} queries, {stats['total_tokens']} tokens")
        print(f"   Average tokens per query: {stats['average_tokens_per_query']:.2f}")
        
        # Finalize session
        logger.finalize_session()
        print("SUCCESS: Session finalized")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Session logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_tracking():
    """Test LLM call tracking"""
    print("\n" + "=" * 60)
    print("TESTING LLM CALL TRACKING")
    print("=" * 60)
    
    try:
        from src.core.agent.llm_tracker import LLMCallTracker, get_global_tracker
        import tiktoken
        
        # Test token encoder
        encoder = tiktoken.get_encoding("cl100k_base")
        test_text = "This is a test prompt for LLM token counting."
        tokens = len(encoder.encode(test_text))
        print(f"SUCCESS: Direct token counting: '{test_text}' = {tokens} tokens")
        
        # Test LLM call tracker
        tracker = LLMCallTracker()
        print("SUCCESS: LLM call tracker initialized")
        
        # Simulate LLM calls
        mock_calls = [
            {
                "timestamp": datetime.now().isoformat(),
                "call_type": "clarification",
                "model": "gpt-4o-mini",
                "prompt": "Analyze user query and determine routing",
                "response": '{"is_greeting": false, "is_safe": true}',
                "input_tokens": 25,
                "output_tokens": 12,
                "total_tokens": 37,
                "duration_seconds": 0.85,
                "llm_name": "test_clarifier"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "call_type": "sql_generation", 
                "model": "gpt-4o-mini",
                "prompt": "Generate SQL for user query about recipes",
                "response": "SELECT DISTINCT SpecDescription FROM Specifications WHERE...",
                "input_tokens": 150,
                "output_tokens": 45,
                "total_tokens": 195,
                "duration_seconds": 1.23,
                "llm_name": "test_sql_generator"
            }
        ]
        
        # Test stats calculation
        total_tokens = sum(call["total_tokens"] for call in mock_calls)
        total_calls = len(mock_calls)
        print(f"SUCCESS: Mock LLM tracking: {total_calls} calls, {total_tokens} tokens")
        
        # Test global tracker
        global_tracker = get_global_tracker()
        stats = global_tracker.get_session_stats()
        print(f"SUCCESS: Global tracker ready: {stats['total_calls']} calls tracked")
        
        return True
        
    except Exception as e:
        print(f"ERROR: LLM tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_components():
    """Test API components without starting server"""
    print("\n" + "=" * 60)
    print("TESTING API COMPONENTS")
    print("=" * 60)
    
    try:
        # Test configuration loading
        from src.api.main_app import CONFIG, ENGINE, inspector
        print(f"SUCCESS: Configuration loaded: {len(CONFIG)} sections")
        
        # Test database connection
        tables = inspector.get_table_names()
        print(f"SUCCESS: Database connected: {len(tables)} tables found")
        for table in tables:
            columns = inspector.get_columns(table)
            print(f"   - {table}: {len(columns)} columns")
        
        # Test FastAPI app
        from src.api.main_app import app
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"SUCCESS: FastAPI app ready: {len(routes)} routes configured")
        important_routes = [r for r in routes if r in ['/', '/chat', '/health', '/session/stats', '/upload']]
        print(f"   Important routes available: {important_routes}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: API components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_system():
    """Test memory system"""
    print("\n" + "=" * 60)
    print("TESTING MEMORY SYSTEM")
    print("=" * 60)
    
    try:
        from src.core.agent.memory import ConversationVectorMemory
        
        # Test memory initialization
        memory = ConversationVectorMemory(max_turns=5)
        print(f"SUCCESS: Memory initialized: max_turns={memory.max_turns}")
        
        # Test basic functionality (without OpenAI API)
        print("SUCCESS: Memory structure ready for conversation tracking")
        print(f"   Current history length: {len(memory.history)}")
        print(f"   Default recent history: {memory.default_recent_history}")
        print(f"   Optional retrieval count: {memory.optional_retrieval_count}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Memory system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_interface():
    """Test Streamlit interface components"""
    print("\n" + "=" * 60)
    print("TESTING STREAMLIT INTERFACE")
    print("=" * 60)
    
    try:
        from src.ui.streamlit_chat import CONFIG, API_BASE_URL
        
        print(f"SUCCESS: Streamlit config loaded")
        print(f"   API Base URL: {API_BASE_URL}")
        print(f"   Config sections: {list(CONFIG.keys())}")
        
        # Test streamlit import
        import streamlit as st
        print(f"SUCCESS: Streamlit available: version {st.__version__}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Streamlit interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("NLQ PLM SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now().isoformat()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 80)
    
    tests = [
        ("Session Logging", test_session_logging),
        ("LLM Tracking", test_llm_tracking), 
        ("API Components", test_api_components),
        ("Memory System", test_memory_system),
        ("Streamlit Interface", test_streamlit_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"PASS: {test_name} test PASSED")
            else:
                print(f"FAIL: {test_name} test FAILED")
        except Exception as e:
            print(f"CRASH: {test_name} test CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED! System is ready for use.")
        print("\nTo start the system:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python src/api/main_app.py")
        print("3. Or run: streamlit run src/ui/streamlit_chat.py")
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now().isoformat()}")
    
    # Show sample log files created
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.startswith("log_") and f.endswith(".log")]
        if log_files:
            print(f"\n📁 Session log files created in /{logs_dir}/:")
            for log_file in sorted(log_files)[-3:]:  # Show last 3
                print(f"   - {log_file}")

if __name__ == "__main__":
    main()