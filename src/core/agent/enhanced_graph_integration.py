# src/core/agent/enhanced_graph_integration.py

"""
Enhanced integration of multi-retrieval system with the existing graph workflow.
This module provides the enhanced retrieval functionality while maintaining 
compatibility with the existing graph.py structure.
"""

import logging
from typing import Dict, List, Optional, Any

from .query_classifier import get_query_classifier, QueryClassification
from ..data_processing.multi_retrieval_system import (
    MultiRetrievalSystem, 
    MultiRetrievalContext, 
    RetrievalType
)
from ..agent.query_analyzer import IntelligentQueryAnalyzer, QueryAnalysis
from ..integration.multi_retrieval_integration import EnhancedRetrievalOrchestrator

logger = logging.getLogger(__name__)


class EnhancedRAGNode:
    """Enhanced RAG node that uses multiple specialized retrievers."""
    
    def __init__(self, engine, schema_data: Optional[Dict] = None):
        self.engine = engine
        self.schema_data = schema_data
        self.orchestrator = EnhancedRetrievalOrchestrator(engine, schema_data)
        self.query_classifier = get_query_classifier()
        self.query_analyzer = IntelligentQueryAnalyzer(schema_data)
        
    def enhanced_rag_retrieval(self, state) -> Dict[str, Any]:
        """Enhanced RAG retrieval with multiple specialized systems."""
        try:
            # Get the query (clarified or original)
            query = state.clarified_query or state.user_query
            
            # Classify the query
            classification = self.query_classifier.classify_query(query)
            
            # Analyze query for retrieval strategy
            query_analysis = self.query_analyzer.analyze_query(query)
            
            # Determine token budget based on complexity
            base_budget = self._calculate_token_budget(classification, query_analysis)
            
            # Get enhanced context using multiple retrievers
            enhanced_context, metadata = self.orchestrator.get_enhanced_context(
                query, max_context_length=base_budget
            )
            
            # Add conversation context if available
            conversation_context = self._build_conversation_context(state)
            
            # Combine all contexts
            final_context = self._combine_contexts(enhanced_context, conversation_context, metadata)
            
            # Update state with enhanced information
            enhanced_state = {
                'rag_context': final_context,
                'retrieval_metadata': metadata,
                'query_classification': classification,
                'query_analysis': query_analysis,
                'enhanced_retrieval_used': True
            }
            
            # Log retrieval information
            logger.info(f"Enhanced retrieval completed: {metadata.get('reasoning', 'No reasoning available')}")
            logger.info(f"Token usage: {metadata.get('token_count', 0)} tokens")
            logger.info(f"Retrieval strategy: {metadata.get('retrieval_strategy', [])}")
            
            return enhanced_state
            
        except Exception as e:
            logger.error(f"Enhanced RAG retrieval failed: {e}")
            
            # Fallback to basic retrieval
            return self._fallback_retrieval(state)
    
    def _calculate_token_budget(self, classification: QueryClassification, 
                              query_analysis: QueryAnalysis) -> int:
        """Calculate appropriate token budget based on query characteristics."""
        base_budget = 3000
        
        # Adjust based on complexity
        if classification.complexity.value == 'complex':
            base_budget = 4500
        elif classification.complexity.value == 'simple':
            base_budget = 2000
        
        # Adjust based on query focus
        if query_analysis.primary_focus.value in ['schema_overview', 'mixed']:
            base_budget += 1000
        elif query_analysis.primary_focus.value in ['column_details', 'data_values']:
            base_budget += 500
        
        # Adjust based on mentioned entities
        if len(query_analysis.mentioned_tables) > 3:
            base_budget += 800
        if len(query_analysis.mentioned_columns) > 5:
            base_budget += 600
        
        return min(6000, base_budget)
    
    def _build_conversation_context(self, state) -> str:
        """Build conversation context from state."""
        conversation_parts = []
        
        # Add similar query context if available
        if hasattr(state, 'similar_query_context'):
            similar = state.similar_query_context
            if similar and isinstance(similar, dict):
                conversation_parts.append(
                    f"Similar Previous Query (similarity: {similar.get('similarity', 0.0):.2f}):"
                )
                conversation_parts.append(f"Question: {similar.get('query', 'N/A')}")
                if similar.get('sql'):
                    conversation_parts.append(f"Previous SQL: {similar['sql']}")
                conversation_parts.append("Use this as inspiration but adapt for the current query.")
        
        # Add any other conversation context
        if hasattr(state, 'conversation_history') and state.conversation_history:
            conversation_parts.append("\nRecent conversation:")
            for entry in state.conversation_history[-3:]:  # Last 3 entries
                if isinstance(entry, dict):
                    conversation_parts.append(f"Q: {entry.get('query', '')}")
                    if entry.get('response'):
                        conversation_parts.append(f"A: {entry['response'][:200]}...")
        
        return "\n".join(conversation_parts) if conversation_parts else ""
    
    def _combine_contexts(self, enhanced_context: str, conversation_context: str, 
                         metadata: Dict) -> str:
        """Combine enhanced context with conversation context."""
        context_parts = []
        
        # Add enhanced retrieval context
        if enhanced_context:
            context_parts.append("=== ENHANCED DATABASE CONTEXT ===")
            context_parts.append(enhanced_context)
        
        # Add conversation context if available
        if conversation_context:
            context_parts.append("\n=== CONVERSATION CONTEXT ===")
            context_parts.append(conversation_context)
        
        # Add retrieval metadata as context hints
        if metadata.get('mentioned_tables'):
            context_parts.append(f"\n=== QUERY ANALYSIS ===")
            context_parts.append(f"Focus: {metadata.get('focus_areas', 'general')}")
            context_parts.append(f"Mentioned tables: {', '.join(metadata['mentioned_tables'])}")
            if metadata.get('mentioned_columns'):
                context_parts.append(f"Mentioned columns: {', '.join(metadata['mentioned_columns'])}")
        
        return "\n".join(context_parts)
    
    def _fallback_retrieval(self, state) -> Dict[str, Any]:
        """Fallback to basic retrieval if enhanced system fails."""
        try:
            from ..data_processing.vectors import combine_retriever_results
            from .graph import APP_STATE
            
            query = state.clarified_query or state.user_query
            
            # Use existing retrieval system
            if "retrievers" in APP_STATE:
                rag_context = combine_retriever_results(
                    APP_STATE["retrievers"], query, max_context_length=3000
                )
            else:
                rag_context = "Schema information temporarily unavailable."
            
            return {
                'rag_context': rag_context,
                'retrieval_metadata': {'fallback_used': True},
                'enhanced_retrieval_used': False
            }
            
        except Exception as e:
            logger.error(f"Fallback retrieval also failed: {e}")
            return {
                'rag_context': "Schema information temporarily unavailable.",
                'retrieval_metadata': {'error': str(e)},
                'enhanced_retrieval_used': False
            }


def create_enhanced_rag_node(engine, schema_data: Optional[Dict] = None) -> EnhancedRAGNode:
    """Factory function to create enhanced RAG node."""
    return EnhancedRAGNode(engine, schema_data)


def integrate_with_existing_graph():
    """Integration function to enhance existing graph workflow."""
    
    def enhanced_rag_retrieval_wrapper(state, engine, schema_data):
        """Wrapper function for existing graph integration."""
        enhanced_node = create_enhanced_rag_node(engine, schema_data)
        return enhanced_node.enhanced_rag_retrieval(state)
    
    return enhanced_rag_retrieval_wrapper


# Demonstration of how to use the enhanced system
def demonstrate_multi_retrieval_capabilities(engine, schema_data: Optional[Dict] = None):
    """Demonstrate the capabilities of the multi-retrieval system."""
    
    enhanced_node = create_enhanced_rag_node(engine, schema_data)
    
    # Test queries that would benefit from different retrieval strategies
    test_queries = [
        {
            "query": "What columns are available in the recipes table?",
            "expected_focus": "column_details",
            "expected_strategy": ["column_descriptions"]
        },
        {
            "query": "What tables contain information about ingredients?",
            "expected_focus": "table_structure", 
            "expected_strategy": ["table_descriptions"]
        },
        {
            "query": "Show me the database schema and how tables are related",
            "expected_focus": "schema_overview",
            "expected_strategy": ["hybrid"]
        },
        {
            "query": "What are the possible values in the status column?",
            "expected_focus": "data_values",
            "expected_strategy": ["column_descriptions"]
        }
    ]
    
    print("=== MULTI-RETRIEVAL SYSTEM DEMONSTRATION ===\n")
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"Test Case {i}: {test_case['query']}")
        print("-" * 60)
        
        # Create mock state
        class MockState:
            def __init__(self, query):
                self.user_query = query
                self.clarified_query = None
        
        state = MockState(test_case['query'])
        
        # Run enhanced retrieval
        result = enhanced_node.enhanced_rag_retrieval(state)
        
        # Display results
        print(f"Expected Focus: {test_case['expected_focus']}")
        print(f"Expected Strategy: {test_case['expected_strategy']}")
        
        if 'query_analysis' in result:
            analysis = result['query_analysis']
            print(f"Detected Focus: {analysis.primary_focus.value}")
            print(f"Detected Strategy: {[rt.value for rt in analysis.retrieval_strategy]}")
            print(f"Confidence: {analysis.confidence:.2f}")
        
        if 'retrieval_metadata' in result:
            metadata = result['retrieval_metadata']
            print(f"Token Count: {metadata.get('token_count', 'N/A')}")
            print(f"Reasoning: {metadata.get('reasoning', 'N/A')}")
        
        print(f"Context Length: {len(result.get('rag_context', ''))}")
        print("=" * 60)
        print()


# Configuration for multi-retrieval system
MULTI_RETRIEVAL_CONFIG = {
    "column_weight": 0.6,
    "table_weight": 0.4,
    "default_token_budget": 3000,
    "max_token_budget": 6000,
    "confidence_threshold": 0.7,
    "enable_fallback": True,
    "enable_caching": True,
    "log_retrieval_stats": True
}
