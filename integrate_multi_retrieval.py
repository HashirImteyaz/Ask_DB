# integrate_multi_retrieval.py

"""
Integration script to enable multiple retrieval systems in the existing Ask_DB application.
Run this script to upgrade your current system with specialized column and table retrievers.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_config_for_multi_retrieval():
    """Update configuration to enable multi-retrieval system."""
    
    config_path = Path("config.json")
    
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Add multi-retrieval configuration
    config["multi_retrieval"] = {
        "enabled": True,
        "default_token_budget": 4000,
        "column_weight": 0.6,
        "table_weight": 0.4,
        "max_tokens": 6000,
        "use_cache": True,
        "embedding_model": "text-embedding-3-small",
        "chunk_sizes": {
            "column": 500,
            "table": 800,
            "hybrid": 600
        },
        "top_k_limits": {
            "column": 8,
            "table": 5,
            "hybrid": 6
        },
        "confidence_threshold": 0.7,
        "enable_fallback": True,
        "log_stats": True
    }
    
    # Add query analysis configuration
    config["query_analysis"] = {
        "enabled": True,
        "use_intelligent_routing": True,
        "confidence_threshold": 0.6
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("Configuration updated for multi-retrieval system")


def create_example_usage():
    """Create example usage script."""
    
    example_script = '''# example_multi_retrieval_usage.py

"""
Example usage of the multiple retrieval system.
This demonstrates how to use both column and table retrievers.
"""

import sqlite3
from sqlalchemy import create_engine
from src.core.data_processing.multi_retrieval_system import (
    create_multi_retrieval_system,
    MultiRetrievalContext,
    RetrievalType
)
from src.core.agent.query_analyzer import analyze_query_for_retrieval
from src.core.integration.multi_retrieval_integration import (
    create_enhanced_retrieval_orchestrator
)

# Database setup
DB_PATH = "DATA/plm_updated.db"  # Update with your database path
engine = create_engine(f"sqlite:///{DB_PATH}")

# Load schema data (if available)
schema_data = {}
try:
    import json
    with open("src/config/schema_description.json", 'r') as f:
        schema_data = json.load(f)
except FileNotFoundError:
    print("Schema description file not found, using basic schema")

def example_column_focused_query():
    """Example: Query focused on column information."""
    
    query = "What columns are available in the specifications table and what do they contain?"
    
    print(f"Query: {query}")
    print("=" * 60)
    
    # Analyze the query
    analysis = analyze_query_for_retrieval(query, schema_data)
    print(f"Detected Focus: {analysis.primary_focus.value}")
    print(f"Retrieval Strategy: {[rt.value for rt in analysis.retrieval_strategy]}")
    print(f"Mentioned Tables: {analysis.mentioned_tables}")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    # Create enhanced orchestrator
    orchestrator = create_enhanced_retrieval_orchestrator(engine, schema_data)
    
    # Get enhanced context
    context, metadata = orchestrator.get_enhanced_context(query, max_context_length=3000)
    
    print("\\nRetrieved Context:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    print(f"\\nMetadata:")
    print(f"- Token Count: {metadata.get('token_count', 'N/A')}")
    print(f"- Strategy Used: {metadata.get('retrieval_strategy', [])}")
    print(f"- Reasoning: {metadata.get('reasoning', 'N/A')}")


def example_table_focused_query():
    """Example: Query focused on table structure."""
    
    query = "What tables are available in the database and how are they related?"
    
    print(f"\\nQuery: {query}")
    print("=" * 60)
    
    # Analyze the query
    analysis = analyze_query_for_retrieval(query, schema_data)
    print(f"Detected Focus: {analysis.primary_focus.value}")
    print(f"Retrieval Strategy: {[rt.value for rt in analysis.retrieval_strategy]}")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    # Create enhanced orchestrator
    orchestrator = create_enhanced_retrieval_orchestrator(engine, schema_data)
    
    # Get enhanced context
    context, metadata = orchestrator.get_enhanced_context(query, max_context_length=3000)
    
    print("\\nRetrieved Context:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)


def example_hybrid_query():
    """Example: Complex query requiring both column and table information."""
    
    query = "Show me all recipe ingredients with their descriptions and explain the database structure"
    
    print(f"\\nQuery: {query}")
    print("=" * 60)
    
    # Analyze the query
    analysis = analyze_query_for_retrieval(query, schema_data)
    print(f"Detected Focus: {analysis.primary_focus.value}")
    print(f"Retrieval Strategy: {[rt.value for rt in analysis.retrieval_strategy]}")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    # Create enhanced orchestrator
    orchestrator = create_enhanced_retrieval_orchestrator(engine, schema_data)
    
    # Get enhanced context
    context, metadata = orchestrator.get_enhanced_context(query, max_context_length=4000)
    
    print("\\nRetrieved Context:")
    print("-" * 40)
    print(context[:600] + "..." if len(context) > 600 else context)


if __name__ == "__main__":
    print("MULTIPLE RETRIEVAL SYSTEM EXAMPLES")
    print("=" * 50)
    
    try:
        example_column_focused_query()
        example_table_focused_query() 
        example_hybrid_query()
        
        print("\\n" + "=" * 50)
        print("Examples completed successfully!")
        print("The system can now intelligently route queries to appropriate retrievers:")
        print("- Column-focused queries -> Column Description Retriever")
        print("- Table-focused queries -> Table Description Retriever") 
        print("- Complex queries -> Hybrid Retriever (both)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your database path and schema are correctly configured.")
'''
    
    with open("example_multi_retrieval_usage.py", 'w', encoding='utf-8') as f:
        f.write(example_script)
    
    logger.info("Created example usage script: example_multi_retrieval_usage.py")


def create_integration_test():
    """Create integration test script."""
    
    test_script = '''# test_multi_retrieval_integration.py

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
    
    print("\\nTesting End-to-End Retrieval...")
    
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
    
    print("\\n" + "=" * 50)
    print("Integration test completed!")
'''
    
    with open("test_multi_retrieval_integration.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    logger.info("Created integration test script: test_multi_retrieval_integration.py")


def update_requirements():
    """Update requirements for multi-retrieval system."""
    
    additional_requirements = [
        "# Multi-retrieval system dependencies",
        "llama-index-core>=0.10.0",
        "llama-index-embeddings-openai>=0.1.0", 
        "sentence-transformers>=2.2.0",
        "# Additional dependencies for enhanced retrieval",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0"
    ]
    
    # Update requirements.txt
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, 'r') as f:
            existing_reqs = f.read()
        
        # Add new requirements if not already present
        with open(req_file, 'a') as f:
            f.write("\\n\\n" + "\\n".join(additional_requirements))
    
    logger.info("Updated requirements.txt with multi-retrieval dependencies")


def main():
    """Main integration function."""
    
    print("INTEGRATING MULTIPLE RETRIEVAL SYSTEM")
    print("=" * 50)
    
    try:
        # Step 1: Update configuration
        logger.info("Step 1: Updating configuration...")
        update_config_for_multi_retrieval()
        
        # Step 2: Update requirements
        logger.info("Step 2: Updating requirements...")
        update_requirements()
        
        # Step 3: Create example usage
        logger.info("Step 3: Creating example usage script...")
        create_example_usage()
        
        # Step 4: Create integration test
        logger.info("Step 4: Creating integration test...")
        create_integration_test()
        
        print("\\n" + "=" * 50)
        print("INTEGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\\nNext Steps:")
        print("1. Install additional dependencies: pip install -r requirements.txt")
        print("2. Run integration test: python test_multi_retrieval_integration.py")
        print("3. Try examples: python example_multi_retrieval_usage.py")
        print("4. The system now supports:")
        print("   - Column Description Retriever: For column-focused queries")
        print("   - Table Description Retriever: For table structure queries") 
        print("   - Hybrid Retriever: For complex queries requiring both")
        print("   - Intelligent Query Analysis: Automatically selects best retriever")
        
        print("\\nHow it works:")
        print("- User queries are analyzed to determine focus (columns vs tables)")
        print("- Appropriate specialized retrievers are selected automatically")  
        print("- Results are combined intelligently based on query complexity")
        print("- Fallback to existing system if enhanced retrieval fails")
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        print("\\nIntegration failed. Please check the error logs and try again.")


if __name__ == "__main__":
    main()
