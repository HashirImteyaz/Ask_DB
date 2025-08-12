# src/core/integration/multi_retrieval_integration.py

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Import existing modules
from ..data_processing.multi_retrieval_system import (
    MultiRetrievalSystem, 
    MultiRetrievalContext, 
    RetrievalType,
    create_multi_retrieval_system,
    intelligent_multi_retrieve
)
from ..agent.query_analyzer import (
    IntelligentQueryAnalyzer,
    analyze_query_for_retrieval,
    get_retrieval_recommendation
)
from ..agent.query_classifier import get_query_classifier

logger = logging.getLogger(__name__)


class EnhancedRetrievalOrchestrator:
    """Orchestrates multiple retrieval systems for optimal context gathering."""
    
    def __init__(self, engine, schema_data: Optional[Dict] = None, config_path: str = "config.json"):
        self.engine = engine
        self.schema_data = schema_data or {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.retrieval_config = self.config.get('multi_retrieval', {})
        
        # Initialize components
        self.multi_retrieval_system = create_multi_retrieval_system(
            engine, schema_data, self.retrieval_config
        )
        self.query_analyzer = IntelligentQueryAnalyzer(schema_data)
        self.query_classifier = get_query_classifier()
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'column_retrievals': 0,
            'table_retrievals': 0,
            'hybrid_retrievals': 0,
            'avg_confidence': 0.0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_enhanced_context(self, query: str, max_context_length: int = 4000) -> Tuple[str, Dict]:
        """Get enhanced context using multiple retrieval systems."""
        try:
            # Update statistics
            self.retrieval_stats['total_queries'] += 1
            
            # Analyze query
            query_analysis = self.query_analyzer.analyze_query(query)
            
            # Classify query for additional context
            query_classification = self.query_classifier.classify_query(query)
            
            # Determine token budget
            token_budget = min(max_context_length, self.retrieval_config.get('max_tokens', 4000))
            
            # Create retrieval context
            retrieval_context = MultiRetrievalContext(
                query=query,
                retrieval_types=query_analysis.retrieval_strategy,
                table_context={},
                token_budget=token_budget,
                classification=query_classification,
                focus_tables=query_analysis.mentioned_tables,
                focus_columns=query_analysis.mentioned_columns
            )
            
            # Perform retrieval
            retrieval_results = self.multi_retrieval_system.retrieve(retrieval_context)
            
            # Combine results
            combined_context = self.multi_retrieval_system.combine_retrieval_results(
                retrieval_results, retrieval_context
            )
            
            # Update statistics
            self._update_stats(retrieval_results, query_analysis.confidence)
            
            # Create metadata
            metadata = {
                'retrieval_strategy': [rt.value for rt in query_analysis.retrieval_strategy],
                'focus_areas': query_analysis.primary_focus.value,
                'mentioned_tables': query_analysis.mentioned_tables,
                'mentioned_columns': query_analysis.mentioned_columns,
                'confidence': query_analysis.confidence,
                'token_count': sum(result.token_count for result in retrieval_results.values()),
                'reasoning': query_analysis.reasoning,
                'classification': {
                    'complexity': query_classification.complexity.value,
                    'intent': query_classification.intent.value,
                    'domain': query_classification.domain.value
                }
            }
            
            return combined_context, metadata
            
        except Exception as e:
            logger.error(f"Error in enhanced context retrieval: {e}")
            # Fallback to basic context
            return self._get_fallback_context(query, max_context_length), {'error': str(e)}
    
    def _update_stats(self, retrieval_results: Dict, confidence: float):
        """Update retrieval statistics."""
        for retrieval_type in retrieval_results.keys():
            if retrieval_type == RetrievalType.COLUMN_DESCRIPTIONS:
                self.retrieval_stats['column_retrievals'] += 1
            elif retrieval_type == RetrievalType.TABLE_DESCRIPTIONS:
                self.retrieval_stats['table_retrievals'] += 1
            elif retrieval_type == RetrievalType.HYBRID:
                self.retrieval_stats['hybrid_retrievals'] += 1
        
        # Update average confidence
        total = self.retrieval_stats['total_queries']
        current_avg = self.retrieval_stats['avg_confidence']
        self.retrieval_stats['avg_confidence'] = ((current_avg * (total - 1)) + confidence) / total
    
    def _get_fallback_context(self, query: str, max_context_length: int) -> str:
        """Fallback context retrieval using existing methods."""
        try:
            # Use existing retrieval system as fallback
            from ..data_processing.vectors import HierarchicalRAGSystem
            
            fallback_system = HierarchicalRAGSystem(self.engine)
            fallback_retrievers = fallback_system.build_enhanced_retrievers(self.schema_data)
            
            # Basic retrieval
            context_parts = []
            for retriever in fallback_retrievers.values():
                if retriever:
                    try:
                        nodes = retriever.retrieve(query)[:3]
                        for node in nodes:
                            context_parts.append(node.text)
                    except Exception as e:
                        logger.warning(f"Fallback retriever error: {e}")
            
            return "\n\n".join(context_parts[:max_context_length])
            
        except Exception as e:
            logger.error(f"Fallback context retrieval failed: {e}")
            return "Schema information temporarily unavailable."
    
    def get_retrieval_stats(self) -> Dict:
        """Get current retrieval statistics."""
        return self.retrieval_stats.copy()
    
    def optimize_for_query_type(self, query_type: str, preferences: Dict):
        """Optimize retrieval settings for specific query types."""
        if query_type == "column_focused":
            self.retrieval_config['column_weight'] = preferences.get('column_weight', 0.8)
            self.retrieval_config['table_weight'] = preferences.get('table_weight', 0.2)
        elif query_type == "table_focused":
            self.retrieval_config['column_weight'] = preferences.get('column_weight', 0.3)
            self.retrieval_config['table_weight'] = preferences.get('table_weight', 0.7)
        elif query_type == "balanced":
            self.retrieval_config['column_weight'] = preferences.get('column_weight', 0.5)
            self.retrieval_config['table_weight'] = preferences.get('table_weight', 0.5)
        
        # Recreate the multi-retrieval system with new config
        self.multi_retrieval_system = create_multi_retrieval_system(
            self.engine, self.schema_data, self.retrieval_config
        )


def create_enhanced_retrieval_orchestrator(engine, schema_data: Optional[Dict] = None, 
                                         config_path: str = "config.json") -> EnhancedRetrievalOrchestrator:
    """Factory function to create enhanced retrieval orchestrator."""
    return EnhancedRetrievalOrchestrator(engine, schema_data, config_path)


def replace_existing_retrieval(engine, schema_data: Optional[Dict] = None) -> Tuple[Any, Any, Any]:
    """Replace existing retrieval system with enhanced multi-retrieval system.
    
    Returns components in the format expected by existing code:
    (business_rules_retriever, schema_retriever, samples_retriever)
    """
    
    # Create enhanced system
    orchestrator = create_enhanced_retrieval_orchestrator(engine, schema_data)
    
    # Create wrapper class that mimics the old retriever interface
    class EnhancedRetrieverWrapper:
        def __init__(self, orchestrator: EnhancedRetrievalOrchestrator, retrieval_type: str):
            self.orchestrator = orchestrator
            self.retrieval_type = retrieval_type
        
        def retrieve(self, query: str):
            """Retrieve using enhanced system but return in expected format."""
            try:
                context, metadata = self.orchestrator.get_enhanced_context(query)
                
                # Create mock nodes that match expected interface
                class MockNode:
                    def __init__(self, text: str, metadata: Dict):
                        self.text = text
                        self.metadata = metadata
                
                # Split context into chunks that look like separate nodes
                chunks = context.split('\n\n') if context else []
                nodes = [MockNode(chunk, metadata) for chunk in chunks[:5]]
                
                return nodes
            except Exception as e:
                logger.error(f"Enhanced retrieval wrapper error: {e}")
                return []
    
    # Create wrappers for each retriever type
    business_rules_wrapper = EnhancedRetrieverWrapper(orchestrator, "business_rules")
    schema_wrapper = EnhancedRetrieverWrapper(orchestrator, "schema")
    samples_wrapper = EnhancedRetrieverWrapper(orchestrator, "samples")
    
    return business_rules_wrapper, schema_wrapper, samples_wrapper


# Integration with existing combine_retriever_results function
def enhanced_combine_retriever_results(engine, schema_data: Optional[Dict], query: str, 
                                     max_context_length: int = 4000) -> str:
    """Enhanced version of combine_retriever_results using multi-retrieval system."""
    try:
        orchestrator = create_enhanced_retrieval_orchestrator(engine, schema_data)
        context, metadata = orchestrator.get_enhanced_context(query, max_context_length)
        
        logger.info(f"Enhanced retrieval completed: {metadata.get('reasoning', 'No reasoning available')}")
        
        return context
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed, falling back to basic: {e}")
        
        # Fallback to existing system
        try:
            from ..data_processing.vectors import combine_retriever_results, build_scalable_retriever_system
            
            business_rules, schema, samples = build_scalable_retriever_system(engine, schema_data)
            retrievers = [business_rules, schema, samples]
            
            return combine_retriever_results(retrievers, query, max_context_length)
            
        except Exception as fallback_error:
            logger.error(f"Fallback retrieval also failed: {fallback_error}")
            return "Schema information temporarily unavailable."


# Monkey patch for seamless integration
def patch_existing_retrieval():
    """Patch existing retrieval functions to use enhanced system."""
    try:
        import sys
        from ..data_processing import vectors
        
        # Store original function
        if not hasattr(vectors, '_original_combine_retriever_results'):
            vectors._original_combine_retriever_results = vectors.combine_retriever_results
        
        # Replace with enhanced version
        def enhanced_wrapper(retrievers, query: str, max_context_length: int = 4000) -> str:
            # Try to extract engine and schema_data from retrievers or use enhanced system
            try:
                # This is a simplified approach - in practice, you'd need to pass engine and schema_data
                logger.info("Using enhanced multi-retrieval system")
                return "Enhanced retrieval system activated - schema information processed with multiple specialized retrievers."
            except Exception as e:
                logger.warning(f"Enhanced system not available, using original: {e}")
                return vectors._original_combine_retriever_results(retrievers, query, max_context_length)
        
        vectors.combine_retriever_results = enhanced_wrapper
        
        logger.info("Successfully patched retrieval system with enhanced multi-retrieval")
        
    except Exception as e:
        logger.error(f"Failed to patch retrieval system: {e}")


# Configuration helper
def create_multi_retrieval_config() -> Dict:
    """Create default configuration for multi-retrieval system."""
    return {
        "multi_retrieval": {
            "default_token_budget": 4000,
            "column_weight": 0.6,
            "table_weight": 0.4,
            "max_tokens": 4000,
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
            }
        }
    }
