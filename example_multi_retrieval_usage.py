# example_multi_retrieval_usage.py

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
    
    print("\nRetrieved Context:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    print(f"\nMetadata:")
    print(f"- Token Count: {metadata.get('token_count', 'N/A')}")
    print(f"- Strategy Used: {metadata.get('retrieval_strategy', [])}")
    print(f"- Reasoning: {metadata.get('reasoning', 'N/A')}")


def example_table_focused_query():
    """Example: Query focused on table structure."""
    
    query = "What tables are available in the database and how are they related?"
    
    print(f"\nQuery: {query}")
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
    
    print("\nRetrieved Context:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)


def example_hybrid_query():
    """Example: Complex query requiring both column and table information."""
    
    query = "Show me all recipe ingredients with their descriptions and explain the database structure"
    
    print(f"\nQuery: {query}")
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
    
    print("\nRetrieved Context:")
    print("-" * 40)
    print(context[:600] + "..." if len(context) > 600 else context)


if __name__ == "__main__":
    print("MULTIPLE RETRIEVAL SYSTEM EXAMPLES")
    print("=" * 50)
    
    try:
        example_column_focused_query()
        example_table_focused_query() 
        example_hybrid_query()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("The system can now intelligently route queries to appropriate retrievers:")
        print("- Column-focused queries -> Column Description Retriever")
        print("- Table-focused queries -> Table Description Retriever") 
        print("- Complex queries -> Hybrid Retriever (both)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your database path and schema are correctly configured.")
