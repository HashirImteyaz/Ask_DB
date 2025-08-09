#!/usr/bin/env python3
"""
Test script for the three final NLQ PLM features:
1. Human-in-the-Loop (HITL)
2. Self-Correction Mechanism  
3. Result Pagination

This script tests the integration of all features to ensure they work correctly.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.agent.query_classifier import QueryClassifier
from src.core.agent.self_corrector import SelfCorrectingAgent
from src.core.agent.sql_logic import execute_sql
from src.core.agent.agent_state import AgentState
from src.core.agent.graph import create_agent_graph
from src.core.agent.llm_tracker import TokenTrackingLLM
from langchain_openai import ChatOpenAI
import sqlite3

# Test configuration
DATABASE_PATH = "DATA/plm_updated.db"
SAMPLE_QUERIES = [
    # Clear query (should not trigger HITL)
    "Show me all employees in the engineering department",
    
    # Ambiguous query (should trigger HITL)
    "Find John's projects",
    
    # Query with large results (should trigger pagination)
    "SELECT * FROM employees",
    
    # Query for self-correction testing
    "Show me the revenue for Q1 2023"
]

class TestFinalFeatures:
    def __init__(self):
        """Initialize the test environment"""
        self.llm = TokenTrackingLLM(
            ChatOpenAI(model="gpt-4", temperature=0),
            project_name="test_final_features"
        )
        self.classifier = QueryClassifier(self.llm)
        self.self_corrector = SelfCorrectingAgent(self.llm)
        self.graph = create_agent_graph(self.llm)
        
    async def test_hitl_feature(self):
        """Test the Human-in-the-Loop feature"""
        print("=" * 60)
        print("TESTING HUMAN-IN-THE-LOOP (HITL) FEATURE")
        print("=" * 60)
        
        # Test clear query (should not trigger HITL)
        clear_query = "Show me all employees in the engineering department"
        print(f"\n1. Testing clear query: '{clear_query}'")
        
        classification = await self.classifier.classify_query(clear_query)
        print(f"   Classification: {classification.category}")
        print(f"   Ambiguity: {classification.ambiguity_analysis.ambiguity_level}")
        print(f"   Should trigger HITL: {classification.ambiguity_analysis.ambiguity_level.name != 'CLEAR'}")
        
        # Test ambiguous query (should trigger HITL)
        ambiguous_query = "Find John's projects"
        print(f"\n2. Testing ambiguous query: '{ambiguous_query}'")
        
        classification = await self.classifier.classify_query(ambiguous_query)
        print(f"   Classification: {classification.category}")
        print(f"   Ambiguity: {classification.ambiguity_analysis.ambiguity_level}")
        print(f"   Should trigger HITL: {classification.ambiguity_analysis.ambiguity_level.name != 'CLEAR'}")
        
        if classification.ambiguity_analysis.suggested_clarifications:
            print("   Clarification options:")
            for i, clarification in enumerate(classification.ambiguity_analysis.suggested_clarifications, 1):
                print(f"     {i}. {clarification}")
        
    async def test_self_correction(self):
        """Test the Self-Correction mechanism"""
        print("\n" + "=" * 60)
        print("TESTING SELF-CORRECTION MECHANISM")
        print("=" * 60)
        
        # Create a test query and mock results
        query = "Show me the top 5 employees by salary"
        mock_sql = "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5"
        mock_results = [
            ("John Doe", 85000),
            ("Jane Smith", 82000),
            ("Bob Johnson", 78000),
            ("Alice Brown", 75000),
            ("Charlie Wilson", 72000)
        ]
        
        print(f"\n1. Testing query: '{query}'")
        print(f"   Generated SQL: {mock_sql}")
        print(f"   Results count: {len(mock_results)}")
        
        # Test self-correction validation
        analysis = await self.self_corrector.validate_results(
            query=query,
            sql=mock_sql,
            results=mock_results,
            database_path=DATABASE_PATH
        )
        
        print(f"\n2. Self-correction analysis:")
        print(f"   Validation result: {analysis.validation_result}")
        print(f"   Confidence score: {analysis.confidence_score}")
        print(f"   Issues found: {len(analysis.issues_found)}")
        
        if analysis.issues_found:
            for issue in analysis.issues_found:
                print(f"     - {issue}")
                
        if analysis.correction_suggestions:
            print("   Correction suggestions:")
            for suggestion in analysis.correction_suggestions:
                print(f"     - {suggestion}")
        
    def test_pagination(self):
        """Test the Result Pagination feature"""
        print("\n" + "=" * 60)
        print("TESTING RESULT PAGINATION")
        print("=" * 60)
        
        # Test pagination with different page sizes
        test_sql = "SELECT * FROM employees"
        
        print(f"\n1. Testing SQL: '{test_sql}'")
        
        # Test first page
        print("\n2. Testing first page (page 1, size 5):")
        try:
            results = execute_sql(
                DATABASE_PATH, 
                test_sql, 
                page_number=1, 
                page_size=5,
                return_total_count=True
            )
            
            print(f"   Results type: {type(results)}")
            if isinstance(results, dict):
                print(f"   Total count: {results.get('total_count', 'N/A')}")
                print(f"   Current page: {results.get('current_page', 'N/A')}")
                print(f"   Page size: {results.get('page_size', 'N/A')}")
                print(f"   Results count: {len(results.get('results', []))}")
                
                if results.get('results'):
                    print("   Sample results:")
                    for i, row in enumerate(results['results'][:3], 1):
                        print(f"     {i}. {row}")
            else:
                print(f"   Results count: {len(results) if results else 0}")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test second page
        print("\n3. Testing second page (page 2, size 5):")
        try:
            results = execute_sql(
                DATABASE_PATH, 
                test_sql, 
                page_number=2, 
                page_size=5,
                return_total_count=True
            )
            
            if isinstance(results, dict):
                print(f"   Current page: {results.get('current_page', 'N/A')}")
                print(f"   Results count: {len(results.get('results', []))}")
            else:
                print(f"   Results count: {len(results) if results else 0}")
                
        except Exception as e:
            print(f"   Error: {e}")

    async def test_integration(self):
        """Test the integration of all features through the agent graph"""
        print("\n" + "=" * 60)
        print("TESTING FULL INTEGRATION")
        print("=" * 60)
        
        # Create agent state
        state = AgentState(
            query="Find projects for Smith",  # Ambiguous query
            classification=None,
            sql_query="",
            results=[],
            answer="",
            graph=None,
            suggested_queries=[],
            cost_info={},
            session_id="test_session",
            conversation_history=[],
            requires_clarification=False,
            clarification_options=[],
            awaiting_user_input=False,
            correction_analysis=None,
            correction_attempt_count=0
        )
        
        print(f"\n1. Testing query: '{state.query}'")
        
        try:
            # Run through the agent graph
            print("\n2. Running through agent graph...")
            result = await self.graph.ainvoke(state)
            
            print("\n3. Final results:")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Requires clarification: {result.get('requires_clarification', False)}")
            print(f"   Clarification options: {len(result.get('clarification_options', []))}")
            print(f"   SQL generated: {bool(result.get('sql_query', ''))}")
            print(f"   Results count: {len(result.get('results', []))}")
            
            if result.get('clarification_options'):
                print("   Clarification options:")
                for option in result['clarification_options']:
                    print(f"     - {option}")
                    
        except Exception as e:
            print(f"   Integration test error: {e}")
    
    def test_database_connection(self):
        """Test database connectivity"""
        print("\n" + "=" * 60)
        print("TESTING DATABASE CONNECTION")
        print("=" * 60)
        
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            print(f"\n1. Database: {DATABASE_PATH}")
            print(f"   Tables found: {len(tables)}")
            for table in tables:
                print(f"     - {table[0]}")
                
            # Test a simple query
            cursor.execute("SELECT COUNT(*) FROM employees")
            employee_count = cursor.fetchone()[0]
            print(f"\n2. Sample query test:")
            print(f"   Total employees: {employee_count}")
            
            conn.close()
            print("   Database connection: ✅ SUCCESS")
            
        except Exception as e:
            print(f"   Database connection: ❌ FAILED - {e}")

async def main():
    """Run all tests"""
    print("FINAL FEATURES INTEGRATION TEST")
    print("Testing Human-in-the-Loop, Self-Correction, and Pagination")
    print("=" * 80)
    
    # Initialize test suite
    test_suite = TestFinalFeatures()
    
    # Run tests
    test_suite.test_database_connection()
    await test_suite.test_hitl_feature()
    await test_suite.test_self_correction()
    test_suite.test_pagination()
    await test_suite.test_integration()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
