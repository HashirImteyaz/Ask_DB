# demo_multi_retrieval.py

"""
Demonstration of the Multiple Retrieval System
This script shows how the system intelligently routes different types of queries
to appropriate specialized retrievers.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.agent.query_analyzer import analyze_query_for_retrieval

def demo_query_analysis():
    """Demonstrate how queries are analyzed and routed to appropriate retrievers."""
    
    print("🔍 MULTIPLE RETRIEVAL SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Test queries demonstrating different retrieval strategies
    test_cases = [
        {
            "query": "What columns are available in the specifications table?",
            "description": "Column-focused query"
        },
        {
            "query": "What tables contain recipe information?",
            "description": "Table-focused query"
        },
        {
            "query": "Show me the database structure and relationships",
            "description": "Schema overview query"
        },
        {
            "query": "What are the possible values in the status column?", 
            "description": "Data values query"
        },
        {
            "query": "How are recipes and ingredients connected?",
            "description": "Relationship query"
        },
        {
            "query": "List all ingredients and their descriptions with table info",
            "description": "Mixed/hybrid query"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 40)
        
        try:
            # Analyze the query
            analysis = analyze_query_for_retrieval(test_case['query'])
            
            # Display analysis results
            print(f"🎯 Detected Focus: {analysis.primary_focus.value.replace('_', ' ').title()}")
            
            if analysis.secondary_focus:
                print(f"🎯 Secondary Focus: {analysis.secondary_focus.value.replace('_', ' ').title()}")
            
            print(f"🔧 Retrieval Strategy: {[rt.value.replace('_', ' ').title() for rt in analysis.retrieval_strategy]}")
            
            if analysis.mentioned_tables:
                print(f"📊 Mentioned Tables: {', '.join(analysis.mentioned_tables)}")
            
            if analysis.mentioned_columns:
                print(f"📈 Mentioned Columns: {', '.join(analysis.mentioned_columns)}")
            
            print(f"📊 Confidence: {analysis.confidence:.1%}")
            
            # Explain the routing decision
            print(f"💡 Reasoning: {analysis.reasoning}")
            
        except Exception as e:
            print(f"❌ Error analyzing query: {e}")
        
        print()
        print("=" * 60)
        print()

def explain_system():
    """Explain how the multiple retrieval system works."""
    
    print("📚 HOW THE MULTIPLE RETRIEVAL SYSTEM WORKS")
    print("=" * 60)
    print()
    
    explanations = [
        {
            "title": "🎯 Query Analysis Phase",
            "description": """
• Analyzes user query to determine primary focus
• Extracts mentioned table and column names  
• Determines query complexity and intent
• Calculates confidence score for routing decisions
            """
        },
        {
            "title": "🔧 Retrieval Strategy Selection", 
            "description": """
• Column Descriptions Retriever: For queries about specific columns, data types, values
• Table Descriptions Retriever: For queries about table structure, schema, relationships  
• Hybrid Retriever: For complex queries requiring both column and table information
• Automatic selection based on query analysis
            """
        },
        {
            "title": "📊 Specialized Retrieval",
            "description": """
• Each retriever has optimized vector embeddings
• Column retriever: Detailed column info, statistics, sample values
• Table retriever: High-level schema, relationships, business context
• Hybrid: Intelligent combination of both approaches
            """
        },
        {
            "title": "🎭 Context Combination",
            "description": """
• Results combined based on query complexity
• Token budget allocated intelligently
• Priority given to most relevant information
• Fallback to existing system if needed
            """
        }
    ]
    
    for explanation in explanations:
        print(explanation["title"])
        print(explanation["description"])
        print()

def show_benefits():
    """Show the benefits of the multiple retrieval system."""
    
    print("✨ BENEFITS OF MULTIPLE RETRIEVAL SYSTEM")
    print("=" * 60)
    print()
    
    benefits = [
        "🎯 **Improved Relevance**: Specialized retrievers return more focused information",
        "⚡ **Better Performance**: Optimized token usage and retrieval efficiency", 
        "🧠 **Intelligent Routing**: Automatic selection of best retrieval strategy",
        "🔄 **Seamless Integration**: Works with existing codebase and interfaces",
        "🛡️ **Reliability**: Fallback mechanisms ensure system always works",
        "📊 **Performance Tracking**: Monitor and optimize retrieval effectiveness"
    ]
    
    for benefit in benefits:
        print(benefit)
        print()

if __name__ == "__main__":
    print("🚀 WELCOME TO THE MULTIPLE RETRIEVAL SYSTEM DEMO")
    print("=" * 60)
    print()
    print("This demonstration shows how your Ask_DB system now uses")
    print("specialized retrievers for different types of queries:")
    print()
    print("• 📊 Column Description Retriever")
    print("• 🏗️  Table Description Retriever") 
    print("• 🔄 Hybrid Retriever")
    print("• 🎯 Intelligent Query Analysis")
    print()
    print("=" * 60)
    print()
    
    try:
        demo_query_analysis()
        explain_system()
        show_benefits()
        
        print("🎉 DEMONSTRATION COMPLETED!")
        print("=" * 60)
        print()
        print("Your Ask_DB system now has sophisticated multiple retrieval capabilities!")
        print("Different types of queries will automatically use the most appropriate retriever.")
        print()
        print("Next steps:")
        print("• The system is ready to use with your existing application")
        print("• Query routing happens automatically based on query analysis")
        print("• Monitor the logs to see which retrievers are being used")
        print("• Adjust configuration in config.json if needed")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure the system is properly configured and dependencies are installed.")
