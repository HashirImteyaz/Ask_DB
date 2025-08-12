# test_multi_retrieval_integration.py

"""
Test script to verify multi-retrieval system integration.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_components():
    """Test individual components."""
    
    print("Testing Multi-Retrieval System Components...")
    
    # Test 1: Multi-retrieval system creation
    try:
        from src.core.data_processing.multi_retrieval_system import create_multi_retrieval_system
        engine = create_engine("sqlite:///DATA/plm_updated.db")
        
        system = create_multi_retrieval_system(engine)
        print("✓ Multi-retrieval system created successfully")
        
    except Exception as e:
        print(f"✗ Multi-retrieval system creation failed: {e}")
    
    # Test 2: Query analyzer
    try:
        from src.core.agent.query_analyzer import analyze_query_for_retrieval
        
        analysis = analyze_query_for_retrieval("What columns are in the table?")
        print(f"✓ Query analyzer working - detected focus: {analysis.primary_focus.value}")
        
    except Exception as e:
        print(f"✗ Query analyzer failed: {e}")
    
    # Test 3: Enhanced orchestrator
    try:
        from src.core.integration.multi_retrieval_integration import create_enhanced_retrieval_orchestrator
        
        orchestrator = create_enhanced_retrieval_orchestrator(engine)
        print("✓ Enhanced orchestrator created successfully")
        
    except Exception as e:
        print(f"✗ Enhanced orchestrator creation failed: {e}")
    
    # Test 4: Integration with existing system
    try:
        from src.core.agent.enhanced_graph_integration import create_enhanced_rag_node
        
        node = create_enhanced_rag_node(engine)
        print("✓ Enhanced RAG node created successfully")
        
    except Exception as e:
        print(f"✗ Enhanced RAG node creation failed: {e}")


def test_end_to_end():
    """Test end-to-end retrieval."""
    
    print("\nTesting End-to-End Retrieval...")
    
    try:
        from src.core.integration.multi_retrieval_integration import enhanced_combine_retriever_results
        from sqlalchemy import create_engine
        
        engine = create_engine("sqlite:///DATA/plm_updated.db")
        
        # Test different query types
        test_queries = [
            "What columns are available?",
            "What tables exist?", 
            "Show me database structure"
        ]
        
        for query in test_queries:
            try:
                context = enhanced_combine_retriever_results(engine, {}, query, 1000)
                print(f"✓ Query '{query}' - Context length: {len(context)}")
            except Exception as e:
                print(f"✗ Query '{query}' failed: {e}")
                
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")


if __name__ == "__main__":
    print("MULTI-RETRIEVAL SYSTEM INTEGRATION TEST")
    print("=" * 50)
    
    test_components()
    test_end_to_end()
    
    print("\n" + "=" * 50)
    print("Integration test completed!")
