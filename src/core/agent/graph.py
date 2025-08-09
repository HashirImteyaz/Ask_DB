import json
from typing import Literal, Dict, Tuple, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.caches import BaseCache
from langchain_openai import ChatOpenAI
from pandas import DataFrame
from langgraph.graph import StateGraph, END, START  # Add START here
# Rebuild ChatOpenAI to fix Pydantic issue
try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass

# Import enhanced components
from .agent_state import AgentState
from .prompts import (
    CLARIFIER_PROMPT, 
    QUERY_REPHRASE_PROMPT, 
    SQL_GENERATION_PROMPT,
    QUERY_COMPLEXITY_ASSESSMENT_PROMPT,
    QUERY_VALIDATION_PROMPT,
    FINAL_ANSWER_PROMPT,
    TABLE_INFO_PROMPT,
    SCHEMA_INTROSPECTION_PROMPT  # ADD THIS LINE
)
from .sql_logic import generate_sql, execute_sql, generate_final_answer, generate_graph, determine_chart_type
from .query_classifier import get_query_classifier, get_model_router, QueryClassification
from .llm_tracker import TokenTrackingLLM, get_global_tracker
from src.core.data_processing.vectors import HierarchicalRAGSystem, RetrievalContext
from src.core.data_processing.vectors import combine_retriever_results

# Load configuration
try:
    with open("config.json", 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}

# Global state for enhanced system
APP_STATE: Dict[str, Any] = {
    "retrievers": None,
    "rag_system": None,
    "query_classifier": None,
    "model_router": None
}

class EnhancedLLMManager:
    """Manages multiple LLM instances with intelligent routing."""
    
    def __init__(self):
        self.models = {}
        self.tracker = get_global_tracker()
        
    def get_llm(self, model_name: str, temperature: float = 0, **kwargs) -> TokenTrackingLLM:
        """Get or create LLM instance with caching."""
        cache_key = f"{model_name}_{temperature}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key not in self.models:
            # Create new LLM instance
            llm = TokenTrackingLLM(
                model=model_name,
                temperature=temperature,
                **kwargs
            )
            
            # Register with global tracker
            self.tracker.register_llm(f"enhanced_{model_name}_{len(self.models)}", llm)
            self.models[cache_key] = llm
            
        return self.models[cache_key]
    
    def route_and_invoke(self, query: str, prompt: str, classification: QueryClassification, call_type: str) -> Any:
        """Route query to appropriate model and invoke."""
        # Get model recommendation
        router = get_model_router()
        routing_info = router.route_query(classification)
        
        # Get appropriate LLM
        model_kwargs = {}
        if routing_info["model"] in ['gpt-4o', 'gpt-4o-mini'] and "json" in call_type.lower():
            model_kwargs["response_format"] = {"type": "json_object"}
        
        llm = self.get_llm(
            routing_info["model"],
            routing_info.get("temperature", 0),
            **model_kwargs
        )
        
        # Invoke with tracking
        response = llm.invoke(prompt, call_type=f"{call_type}_{routing_info['model']}")
        
        # Log routing decision
        print(f"DEBUG: Routed {call_type} to {routing_info['model']} - {routing_info['reasoning']}")
        
        return response

# Global LLM manager
_llm_manager = None

def get_llm_manager() -> EnhancedLLMManager:
    """Get global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = EnhancedLLMManager()
    return _llm_manager

def should_trigger_schema_introspection(user_query: str) -> bool:
    """
    Determine if a query should trigger schema introspection.
    This is automatically handled by complexity assessment, but can be used for manual checks.
    """
    
    # Keywords that often indicate need for schema introspection
    schema_triggers = [
        'join', 'relationship', 'connect', 'related', 'foreign key',
        'multiple tables', 'combine', 'merge', 'across tables',
        'complex query', 'detailed analysis', 'comprehensive'
    ]
    
    query_lower = user_query.lower()
    
    # Check for explicit schema exploration requests
    if any(trigger in query_lower for trigger in schema_triggers):
        return True
    
    # Check for complex analytical patterns
    analytical_patterns = [
        'trend analysis', 'correlation', 'distribution', 'breakdown by',
        'grouped by', 'aggregate', 'summary statistics', 'compare across'
    ]
    
    if any(pattern in query_lower for pattern in analytical_patterns):
        return True
    
    # Multi-word queries with multiple entities often need schema validation
    words = query_lower.split()
    if len(words) > 10 and len([w for w in words if w.isalpha()]) > 7:
        return True
    
    return False

# Enhanced Graph Nodes with Intelligence
def query_complexity_assessment_node(state: AgentState) -> AgentState:
    """Assess query complexity for intelligent routing."""
    
    # Prepare context
    schema_context = ""
    if APP_STATE["rag_system"] and hasattr(APP_STATE["rag_system"], 'db_stats'):
        schema_context = str(APP_STATE["rag_system"].db_stats)
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.chat_history[-3:]])
    
    prompt = QUERY_COMPLEXITY_ASSESSMENT_PROMPT.format(
        user_query=state.clarified_query or state.user_query,
        schema_context=schema_context,
        chat_history=history_str
    )
    
    classification = getattr(state, 'query_classification', None)
    if not classification:
        classifier = get_query_classifier()
        classification = classifier.classify_query(state.user_query)
    
    llm_manager = get_llm_manager()
    
    try:
        response = llm_manager.route_and_invoke(
            state.user_query, prompt, classification, "complexity_assessment_json"
        )
        
        # Parse JSON response
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        content = content.strip()
        
        complexity_data = json.loads(content)
        
        # Store complexity assessment in state
        state.complexity_assessment = complexity_data
        state.processing_strategy = complexity_data.get('recommended_approach', 'enhanced')
        
        print(f"DEBUG: Complexity assessed as {complexity_data.get('overall_complexity', 'moderate')}")
        
    except Exception as e:
        print(f"Warning: Complexity assessment failed: {e}")
        # Fallback to basic assessment
        state.complexity_assessment = {
            "overall_complexity": "moderate",
            "recommended_approach": "enhanced"
        }
    
    return state

def intelligent_query_analysis_node(state: AgentState) -> AgentState:
    """New node: Analyze query with enhanced classification."""
    classifier = get_query_classifier()
    
    # Get table context if available
    table_context = {}
    if APP_STATE["rag_system"] and hasattr(APP_STATE["rag_system"], 'db_stats'):
        table_context = APP_STATE["rag_system"].db_stats
    
    # Classify the query
    classification = classifier.classify_query(state.user_query, context={"table_info": table_context})
    
    # Store classification in state for downstream nodes
    state.query_classification = classification
    state.estimated_cost = classification.estimated_cost
    state.recommended_model = classification.recommended_model
    
    # Check for ambiguity and set HITL flags
    if classification.ambiguity_analysis:
        ambiguity = classification.ambiguity_analysis
        if ambiguity.ambiguity_level.value in ['ambiguous', 'requires_clarification']:
            state.requires_clarification = True
            state.clarification_type = 'ambiguity'
            state.clarification_options = ambiguity.suggested_clarifications
            state.awaiting_user_input = True
            state.clarification_context = {
                'ambiguous_terms': ambiguity.ambiguous_terms,
                'clarification_needed': ambiguity.clarification_needed,
                'confidence': ambiguity.confidence
            }
    
    print(f"DEBUG: Query classified as {classification.complexity.value} {classification.intent.value} query")
    print(f"DEBUG: Estimated cost: {classification.estimated_cost} tokens")
    print(f"DEBUG: Recommended model: {classification.recommended_model}")
    
    if state.requires_clarification:
        print(f"DEBUG: Query requires clarification due to ambiguity")
    
    return state

def enhanced_clarifier_node(state: AgentState) -> AgentState:
    """Enhanced clarifier with intelligent model routing."""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.chat_history[-6:]])
    prompt = CLARIFIER_PROMPT.format(chat_history=history_str, user_query=state.user_query)

    # Use classification for intelligent routing
    classification = getattr(state, 'query_classification', None)
    if not classification:
        # Fallback classification for safety
        classifier = get_query_classifier()
        classification = classifier.classify_query(state.user_query)
    
    # Route to appropriate model
    llm_manager = get_llm_manager()
    response = llm_manager.route_and_invoke(
        state.user_query, 
        prompt, 
        classification, 
        "clarification_json"
    )
    
    try:
        # Clean response content - remove markdown code blocks if present
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.startswith('```'):
            content = content[3:]   # Remove ```
        if content.endswith('```'):
            content = content[:-3]  # Remove closing ```
        content = content.strip()
        
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON from clarifier. Response: {response.content}")
        # Enhanced fallback with better detection
        parsed = {
            "is_greeting": any(word in state.user_query.lower() for word in ["hello", "hi", "hey", "good morning", "good afternoon"]),
            "is_safe": not any(unsafe in state.user_query.upper() for unsafe in ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]),
            "needs_clarification": len(state.user_query.split()) < 3 and "?" in state.user_query,
            "wants_graph": any(word in state.user_query.lower() for word in ["graph", "chart", "plot", "visualiz", "diagram"]),
            "clarification_question": ""
        }
    
    state.is_greeting = parsed.get("is_greeting", False)
    state.is_safe = parsed.get("is_safe", True)
    state.needs_clarification = parsed.get("needs_clarification", False)
    state.wants_graph = parsed.get("wants_graph", False)
    state.clarification_question = parsed.get("clarification_question", "")

    if state.is_greeting:
        state.final_answer = "Hello! I'm your intelligent data analysis assistant. I can help you explore and analyze your data using natural language queries. What would you like to know about your data?"
    elif not state.is_safe:
        state.final_answer = "I'm designed to safely analyze and retrieve data. I can only perform read operations to ensure your data remains secure."
    elif state.needs_clarification:
        state.final_answer = state.clarification_question or "Could you please provide more details about what you're looking for?"
    
    return state

def enhanced_query_rephrase_node(state: AgentState) -> AgentState:
    """
    Enhanced query rephrasing that uses conversation memory and a powerful prompt to resolve context.
    This node now runs BEFORE the clarifier to proactively resolve ambiguity.
    """
    # Use context_history from conversation memory instead of chat_history
    relevant_context = []
    
    if hasattr(state, 'context_history') and state.context_history:
        # Convert context_history to chat-like format for the prompt
        for ctx in state.context_history[:3]:  # Use top 3 most relevant contexts
            if ctx.get('query') and ctx.get('response'):
                relevant_context.append({
                    "role": "user",
                    "content": ctx['query']
                })
                relevant_context.append({
                    "role": "assistant", 
                    "content": ctx['response']
                })
    
    # If no context from memory, fall back to chat_history
    if not relevant_context and state.chat_history:
        relevant_context = state.chat_history[-6:]
    
    # If still no context, skip rephrasing
    if not relevant_context:
        state.clarified_query = state.user_query
        return state

    # Format the conversation context as a JSON string for clearer parsing by the LLM.
    history_str = json.dumps(relevant_context, indent=2)

    prompt = QUERY_REPHRASE_PROMPT.format(
        chat_history=history_str,
        current_query=state.user_query
    )

    classification = state.query_classification or get_query_classifier().classify_query(state.user_query)
    llm_manager = get_llm_manager()
    
    try:
        response = llm_manager.route_and_invoke(
            state.user_query, prompt, classification, "query_rephrase"
        )
        rephrased_query = response.content.strip()

        # Quality check: did rephrasing actually resolve vague references?
        original_lower = state.user_query.lower()
        rephrased_lower = rephrased_query.lower()
        vague_words = ["these", "those", "that", "it", "them"]
        
        # Check if vague words were present in original but resolved in rephrased
        original_has_vague = any(f" {word} " in f" {original_lower} " for word in vague_words)
        rephrased_has_vague = any(f" {word} " in f" {rephrased_lower} " for word in vague_words)
        
        if (rephrased_query and 
            rephrased_query.lower() != state.user_query.lower() and
            len(rephrased_query.split()) > len(state.user_query.split())):
            
            state.clarified_query = rephrased_query
            print(f"DEBUG: Query rephrased: '{state.user_query}' → '{rephrased_query}'")
            
            # If we successfully reduced vague words, mark as successfully clarified
            if original_has_vague and not rephrased_has_vague:
                print("DEBUG: Successfully resolved vague references in query")
                state.requires_clarification = False
                
        else:
            state.clarified_query = state.user_query
            print(f"DEBUG: No significant rephrasing achieved for query: '{state.user_query}'")

    except Exception as e:
        print(f"Warning: Enhanced rephrasing failed: {e}. Falling back to original query.")
        state.clarified_query = state.user_query

    return state

def route_after_rephrase(state: AgentState) -> Literal["generate_clarification", "enhanced_history_check"]:
    """
    Decides whether to ask for clarification AFTER attempting to rephrase the query.
    """
    original_query = state.user_query.lower()
    rephrased_query = state.clarified_query.lower()
    
    # If the queries are significantly different, rephrasing likely succeeded
    if original_query != rephrased_query:
        # Check if the rephrased query is more specific/complete than original
        original_words = set(original_query.split())
        rephrased_words = set(rephrased_query.split())
        
        # If rephrased query has significantly more content, it likely resolved ambiguity
        if len(rephrased_words) > len(original_words) * 1.5:
            print("DEBUG: Rephrasing added significant context. Proceeding to history check.")
            state.requires_clarification = False  # Reset since rephrasing succeeded
            return "enhanced_history_check"
    
    # Check if vague words are still present without resolution
    vague_words = ["these", "those", "that", "it", "them"]
    still_vague = any(f" {word} " in f" {rephrased_query} " for word in vague_words)
    
    # Only ask for clarification if:
    # 1. We originally needed clarification AND
    # 2. The query is still vague AND  
    # 3. No conversation history is available to provide context
    if (state.requires_clarification and still_vague and 
        (not state.context_history or len(state.context_history) == 0)):
        print("DEBUG: Query still ambiguous and no history available. Routing to clarifier.")
        return "generate_clarification"
    
    print("DEBUG: Proceeding to history check - either rephrasing succeeded or context is available.")
    state.requires_clarification = False  # Reset since we're proceeding with resolution
    return "enhanced_history_check"

def enhanced_history_check_node(state: AgentState) -> AgentState:
    """Enhanced history checking with semantic similarity."""
    if state.context_history:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        try:
            # Use more sophisticated similarity matching
            current_query = state.clarified_query or state.user_query
            
            # Prepare documents for similarity comparison
            documents = [current_query]
            context_queries = []
            
            for context in state.context_history:
                context_query = context.get('query', '')
                if context_query:
                    documents.append(context_query)
                    context_queries.append(context)
            
            if len(documents) > 1:
                # Calculate TF-IDF similarity
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Calculate cosine similarities
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                
                # Find best match above threshold
                max_similarity_idx = np.argmax(similarities)
                max_similarity = similarities[max_similarity_idx]
                
                # Higher threshold for reuse but with confidence scoring
                if max_similarity > 0.85:  # Very high similarity
                    best_context = context_queries[max_similarity_idx]
                    previous_response = best_context.get('response', '')
                    
                    state.final_answer = f"{previous_response}\n\n*Retrieved from recent conversation history (similarity: {max_similarity:.2f})*"
                    state.from_history = True
                    state.confidence_score = max_similarity
                    
                    print(f"DEBUG: Reusing historical response with {max_similarity:.2f} similarity")
                    return state
                elif max_similarity > 0.7:  # Good similarity - provide as context
                    # Don't reuse but add as enhanced context for SQL generation
                    best_context = context_queries[max_similarity_idx]
                    state.similar_query_context = {
                        'query': best_context.get('query', ''),
                        'response': best_context.get('response', ''),
                        'sql': best_context.get('sql', ''),
                        'similarity': max_similarity
                    }
                    print(f"DEBUG: Found similar query context with {max_similarity:.2f} similarity")
        
        except Exception as e:
            print(f"Warning: Enhanced history check failed: {e}")
    
    return state

def enhanced_sql_generation_node(state: AgentState) -> AgentState:
    """Enhanced SQL generation with intelligent RAG, schema introspection and new prompt system."""
    if APP_STATE["rag_system"] is None:
        state.final_answer = "The database has not been loaded. Please upload a data file first."
        return state
    
    # Create enhanced retrieval context
    classification = getattr(state, 'query_classification', None)
    if not classification:
        classifier = get_query_classifier()
        classification = classifier.classify_query(state.clarified_query or state.user_query)
    
    # Determine token budget based on query complexity
    base_budget = {
        'simple': 2000,
        'moderate': 3000, 
        'complex': 4500
    }.get(classification.complexity.value, 2500)
    
    # Create retrieval context
    retrieval_context = RetrievalContext(
        query=state.clarified_query or state.user_query,
        classification=classification,
        table_context=getattr(APP_STATE["rag_system"], 'db_stats', {}),
        token_budget=base_budget
    )
    
    # Get intelligent context
    if isinstance(APP_STATE["rag_system"], HierarchicalRAGSystem):
        rag_context = APP_STATE["rag_system"].intelligent_retrieve(retrieval_context)
    else:
        # Fallback to legacy system
        rag_context = combine_retriever_results(APP_STATE["retrievers"], state.clarified_query or state.user_query, base_budget)
    
    # Add similar query context if available
    conversation_context = ""
    if hasattr(state, 'similar_query_context'):
        similar = state.similar_query_context
        if similar and isinstance(similar, dict):
            conversation_context = f"\n\nSimilar Previous Query (similarity: {similar.get('similarity', 0.0):.2f}):\n"
            conversation_context += f"Question: {similar.get('query', 'N/A')}\n"
            if similar.get('sql'):
                conversation_context += f"Previous SQL: {similar['sql']}\n"
            conversation_context += "Use this as inspiration but adapt for the current query."
    
    # NEW: Add schema introspection context for complex queries
    complexity_assessment = getattr(state, 'complexity_assessment', {})
    
    # Check if schema introspection is needed
    needs_schema_introspection = (
        complexity_assessment.get('needs_schema_check', False) or
        complexity_assessment.get('overall_complexity') in ['complex', 'advanced'] or
        classification.complexity.value in ['complex', 'advanced']
    )
    
    schema_context = ""
    if needs_schema_introspection:
        from .sql_logic import perform_schema_introspection
        schema_context = perform_schema_introspection(
            state.clarified_query or state.user_query, 
            complexity_assessment
        )
        print(f"DEBUG: Schema introspection applied for {classification.complexity.value} query")
    
    state.rag_context = rag_context + conversation_context + schema_context
    
    # Use the enhanced SQL generation prompt
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.chat_history[-3:]])
    
    enhanced_sql_prompt = SQL_GENERATION_PROMPT.format(
        business_rules=state.rag_context,
        history=history_str,
        user_query=state.clarified_query or state.user_query
    )
    
    # Route to appropriate model for SQL generation
    llm_manager = get_llm_manager()
    response = llm_manager.route_and_invoke(
        state.user_query, 
        enhanced_sql_prompt, 
        classification, 
        "sql_generation"
    )
    
    state.sql_query = response.content.strip()
    
    # Store schema introspection flag for debugging
    state.schema_introspection_applied = needs_schema_introspection
    
    return state

def enhanced_sql_execution_node(state: AgentState) -> AgentState:
    """Enhanced SQL execution with intelligent error handling."""
    if not state.sql_query:
        # Try to provide more intelligent guidance
        classification = getattr(state, 'query_classification', None)
        if classification and classification.intent.value == 'analysis':
            state.final_answer = "I understand you want to analyze the data, but I need more specific information about what metrics or dimensions you're interested in. Could you provide more details?"
        else:
            state.final_answer = "I couldn't generate a SQL query for your request. Please try rephrasing your question with more specific details about what you're looking for."
        state.needs_clarification = True
        return state
    
    from .sql_logic import retry_sql_generation, generate_error_clarification
    
    # Enhanced retry logic with learning
    max_retries = 3
    current_sql = state.sql_query
    retry_history = []
    
    for attempt in range(max_retries):
        # Use pagination if needed (default values for now, can be enhanced later)
        result = execute_sql(current_sql, page_number=1, page_size=1000, return_total_count=True)
        
        # Extract DataFrame from result if it's a pagination dictionary
        if isinstance(result, dict) and 'data' in result:
            actual_result = result['data']
            # Store pagination info in state if needed (can be enhanced later)
            state.performance_metrics.update({
                'pagination_info': {
                    'page': result.get('page', 1),
                    'total_rows': result.get('total_rows'),
                    'has_more': result.get('has_more', False)
                }
            })
        else:
            actual_result = result
        
        # Enhanced success detection - check if we got a valid result (DataFrame or valid string, not error)
        is_error = (isinstance(actual_result, str) and 
                   ("SQL Error:" in actual_result or "Unexpected Error:" in actual_result or "Error:" in actual_result))
        
        if not is_error:
            # Success! Store the final SQL and result (even if empty DataFrame)
            state.sql_query = current_sql
            state.sql_result = actual_result
            state.retry_count = attempt
            
            # Log successful SQL pattern for learning
            if attempt > 0:
                print(f"DEBUG: SQL succeeded after {attempt + 1} attempts")
            
            return state
        
        # SQL failed - analyze error and retry intelligently
        error_message = actual_result if isinstance(actual_result, str) else str(actual_result)
        retry_history.append({'sql': current_sql, 'error': error_message, 'attempt': attempt + 1})
        
        if attempt < max_retries - 1:
            print(f"DEBUG: SQL attempt {attempt + 1} failed, analyzing error for intelligent retry...")
            
            # Enhanced retry with error pattern analysis
            classification = getattr(state, 'query_classification', None)
            
            # Use appropriate model for retry based on complexity
            retry_sql = retry_sql_generation(
                user_query=state.clarified_query or state.user_query,
                rag_context=state.rag_context,
                previous_sql=current_sql,
                error_msg=result,
                attempt=attempt + 1
            )
            
            if retry_sql and retry_sql != current_sql:
                current_sql = retry_sql
                print(f"DEBUG: Generated new SQL for retry: {retry_sql[:100]}...")
            else:
                print(f"Warning: Retry attempt {attempt + 1} produced same or no SQL")
                break
        else:
            print(f"DEBUG: All {max_retries} attempts failed")
    
    # All retries failed - provide intelligent error handling
    # Apply the same extraction logic in case of final failure
    final_result = result
    if isinstance(result, dict) and 'data' in result:
        final_result = result['data']
    
    state.sql_error = final_result
    state.sql_result = final_result
    state.needs_clarification = True
    state.retry_count = max_retries
    
    # Enhanced error clarification
    classification = getattr(state, 'query_classification', None)
    
    # Analyze error patterns from retry history
    error_pattern = "execution_error"
    if any("no such table" in r['error'].lower() for r in retry_history):
        error_pattern = "table_not_found"
    elif any("no such column" in r['error'].lower() for r in retry_history):
        error_pattern = "column_not_found"
    elif any("syntax error" in r['error'].lower() for r in retry_history):
        error_pattern = "syntax_error"
    
    # Provide targeted guidance based on error pattern and query classification
    if error_pattern == "table_not_found":
        state.final_answer = f"I couldn't find the table referenced in your query. Let me help you explore the available data. What type of information are you looking for?"
    elif error_pattern == "column_not_found":
        state.final_answer = f"I couldn't find the specific column mentioned in your query. Could you describe what data field you're interested in? I can help identify the correct column name."
    elif error_pattern == "syntax_error":
        clarification = generate_error_clarification(state.user_query, current_sql, result)
        state.final_answer = f"I had trouble constructing the SQL query. {clarification}"
    else:
        clarification = generate_error_clarification(state.user_query, current_sql, result)
        state.final_answer = f"After {max_retries} attempts, I couldn't execute your query successfully. {clarification}"
    
    return state

def enhanced_analysis_node(state: AgentState) -> AgentState:
    """Enhanced analysis with intelligent response generation."""
    # Skip analysis if there was an SQL error
    if state.needs_clarification and state.sql_error:
        return state
    
    # Use intelligent model routing for response generation
    classification = getattr(state, 'query_classification', None)
    complexity_level = "moderate"
    
    if classification:
        complexity_level = classification.complexity.value
    elif hasattr(state, 'complexity_assessment'):
        complexity_level = state.complexity_assessment.get('overall_complexity', 'moderate')
    
    # Use the enhanced final answer prompt
    enhanced_final_prompt = FINAL_ANSWER_PROMPT.format(
        user_question=state.clarified_query or state.user_query,
        table_text=str(state.sql_result) if state.sql_result is not None else "No data",
        complexity_level=complexity_level
    )
    
    if classification:
        # Route final answer generation to appropriate model
        llm_manager = get_llm_manager()
        response = llm_manager.route_and_invoke(
            state.user_query,
            enhanced_final_prompt,
            classification,
            "final_answer_generation"
        )
        
        state.final_answer = response.content.strip()
        
        # Add confidence and cost information
        if hasattr(state, 'confidence_score') and state.confidence_score is not None:
            if state.confidence_score < 0.7:
                state.final_answer += f"\n\n*Note: This response has moderate confidence ({state.confidence_score:.2f}). Please verify the results.*"
    else:
        # Fallback to standard generation
        from .sql_logic import generate_final_answer
        state.final_answer = generate_final_answer(
            state.clarified_query or state.user_query, 
            state.sql_result
        )
    
    # Enhanced graph generation (existing code remains the same)
    if state.wants_graph and isinstance(state.sql_result, DataFrame) and not state.sql_result.empty:
        if len(state.sql_result.columns) >= 2:
            state.graph_b64 = generate_graph(state.sql_result, state.user_query or state.clarified_query)
            if state.graph_b64:
                chart_type = determine_chart_type(state.sql_result, state.user_query or state.clarified_query)
                state.final_answer += f"\n\n[CHART] **Enhanced {chart_type.title()} visualization generated!**"
                
                if classification and classification.intent.value in ['analysis', 'trend']:
                    state.final_answer += f" The chart shows patterns relevant to your {classification.intent.value} request."
    
    return state
# HITL and Self-Correction Nodes

def generate_clarification_node(state: AgentState) -> AgentState:
    """Generate clarification questions for ambiguous queries."""
    if not state.requires_clarification:
        return state
    
    # Build clarification message
    if state.clarification_options:
        clarification_msg = "I need some clarification to provide an accurate answer:\n\n"
        
        for i, option in enumerate(state.clarification_options, 1):
            clarification_msg += f"**{option.get('type', 'clarification').replace('_', ' ').title()}:**\n"
            clarification_msg += f"{option.get('message', '')}\n\n"
            
            if option.get('options'):
                clarification_msg += "Please choose one:\n"
                for j, choice in enumerate(option['options'], 1):
                    clarification_msg += f"{j}. {choice}\n"
                clarification_msg += "\n"
        
        clarification_msg += "Please provide the clarification so I can help you better."
        
        state.final_answer = clarification_msg
        state.needs_clarification = True
    else:
        state.final_answer = "I need more information to answer your question accurately. Could you please be more specific?"
        state.needs_clarification = True
    
    return state

def self_correction_node(state: AgentState) -> AgentState:
    """Validate the result and trigger corrections if needed."""
    # Skip if we're already in a correction loop or no SQL result
    if state.correction_attempt_count >= state.max_correction_attempts:
        return state
    
    # Check if sql_result is None, empty, or error string
    if (state.sql_result is None or 
        isinstance(state.sql_result, str) or 
        (hasattr(state.sql_result, 'empty') and state.sql_result.empty)):
        return state
    
    try:
        from .self_corrector import get_self_corrector
        
        corrector = get_self_corrector()
        
        # Get context for validation
        context = {}
        if hasattr(state, 'query_classification'):
            context['classification'] = state.query_classification
        if state.rag_context:
            context['rag_context'] = state.rag_context[:1000]  # Limit context size
        
        # Perform validation
        analysis = corrector.validate_result(
            user_query=state.user_query,
            sql_query=state.sql_query,
            result_data=state.sql_result,
            context=context
        )
        
        # Store analysis
        state.correction_analysis = {
            'validation_result': analysis.validation_result.value,
            'confidence': analysis.confidence,
            'issues_found': analysis.issues_found,
            'suggested_corrections': analysis.suggested_corrections,
            'reasoning': analysis.reasoning
        }
        
        # Decide if retry is needed
        if analysis.should_retry and analysis.validation_result.value in ['invalid', 'suspicious']:
            if analysis.confidence < 0.4:  # Low confidence in current result
                state.self_correction_needed = True
                state.correction_attempt_count += 1
                
                # Add retry instructions to help next SQL generation
                if analysis.retry_instructions:
                    retry_context = f"Previous attempt had issues: {', '.join(analysis.issues_found)}. "
                    retry_context += f"Correction needed: {analysis.retry_instructions}"
                    state.rag_context = (state.rag_context or "") + f"\n\nSELF-CORRECTION: {retry_context}"
                
                print(f"DEBUG: Self-correction triggered. Attempt {state.correction_attempt_count}/{state.max_correction_attempts}")
                print(f"DEBUG: Issues found: {analysis.issues_found}")
        
    except Exception as e:
        print(f"DEBUG: Self-correction failed: {e}")
        # Continue without correction
    
    return state

# Enhanced Routing Logic

def route_after_analysis(state: AgentState) -> Literal["generate_clarification", "enhanced_rephrase", "end_conversation"]:
    """Route after initial query analysis."""
    if state.is_greeting or not state.is_safe:
        return "end_conversation"
    
    # ALWAYS try rephrasing first, even if requires_clarification is True
    # The rephrasing step will decide if clarification is still needed
    return "enhanced_rephrase"

def route_after_enhanced_rephrase(state: AgentState) -> Literal["enhanced_history_check", "generate_clarification"]:
    """Route after query rephrasing - use our smart routing logic."""
    return route_after_rephrase(state)

def route_after_enhanced_history(state: AgentState) -> Literal["end_conversation", "enhanced_sql_generation"]:
    if state.from_history:
        return "end_conversation"
    return "enhanced_sql_generation"

def route_after_sql_execution(state: AgentState) -> Literal["self_correction", "enhanced_analysis"]:
    """Route after SQL execution to decide if self-correction is needed."""
    if state.self_correction_needed:
        return "enhanced_sql_generation"  # Loop back to regenerate SQL
    return "self_correction"  # Check if result needs correction

def route_after_self_correction(state: AgentState) -> Literal["enhanced_sql_generation", "enhanced_analysis"]:
    """Route after self-correction analysis."""
    if state.self_correction_needed and state.correction_attempt_count < state.max_correction_attempts:
        return "enhanced_sql_generation"  # Try again with corrections
    return "enhanced_analysis"  # Proceed with current result

# Build Enhanced Graph

def build_enhanced_agent_graph():
    """Build the enhanced agent graph with intelligence routing."""
    workflow = StateGraph(AgentState)
    
    # Add enhanced nodes
    workflow.add_node("query_analysis", intelligent_query_analysis_node)
    workflow.add_node("assess_complexity", query_complexity_assessment_node)  # RENAMED NODE
    workflow.add_node("generate_clarification", generate_clarification_node)
    workflow.add_node("enhanced_clarifier", enhanced_clarifier_node)
    workflow.add_node("enhanced_rephrase", enhanced_query_rephrase_node)
    workflow.add_node("enhanced_history_check", enhanced_history_check_node)
    workflow.add_node("enhanced_sql_generation", enhanced_sql_generation_node)
    workflow.add_node("enhanced_sql_execution", enhanced_sql_execution_node)
    workflow.add_node("self_correction", self_correction_node)
    workflow.add_node("enhanced_analysis", enhanced_analysis_node)
    
    # Define the enhanced flow
# Define the enhanced flow
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "assess_complexity")  
    workflow.add_edge("assess_complexity", "enhanced_clarifier")
    
    workflow.add_conditional_edges(
        "enhanced_clarifier",
        route_after_analysis,
        {
            "enhanced_rephrase": "enhanced_rephrase", 
            "generate_clarification": "generate_clarification",
            "end_conversation": END
        }
    )
    
    # HITL clarification ends the conversation (waits for user input)
    workflow.add_edge("generate_clarification", END)
    
    workflow.add_conditional_edges(
        "enhanced_rephrase", 
        route_after_enhanced_rephrase,
        {
            "enhanced_history_check": "enhanced_history_check",
            "generate_clarification": "generate_clarification"
        }
    )
    
    workflow.add_conditional_edges(
        "enhanced_history_check",
        route_after_enhanced_history,
        {"enhanced_sql_generation": "enhanced_sql_generation", "end_conversation": END}
    )
    
    workflow.add_edge("enhanced_sql_generation", "enhanced_sql_execution")
    
    workflow.add_conditional_edges(
        "enhanced_sql_execution",
        route_after_sql_execution,
        {"self_correction": "self_correction", "enhanced_analysis": "enhanced_analysis"}
    )
    
    workflow.add_conditional_edges(
        "self_correction",
        route_after_self_correction,
        {"enhanced_sql_generation": "enhanced_sql_generation", "enhanced_analysis": "enhanced_analysis"}
    )
    
    workflow.add_edge("enhanced_analysis", END)
    
    return workflow.compile()

# Initialization function for the enhanced system

def initialize_enhanced_system(engine, schema_data=None):
    """Initialize the enhanced RAG and classification systems."""
    global APP_STATE
    
    try:
        # Initialize RAG system
        print("Initializing HierarchicalRAGSystem in graph.py...")
        rag_system = HierarchicalRAGSystem(engine)
        APP_STATE["rag_system"] = rag_system
        
        print("Building enhanced retrievers...")
        retrievers = rag_system.build_enhanced_retrievers(schema_data)
        APP_STATE["retrievers"] = retrievers
        print(f"Retrievers built successfully: {type(retrievers)}")
        
        # Initialize classifier and router
        print("Initializing query classifier...")
        APP_STATE["query_classifier"] = get_query_classifier()
        
        print("Initializing model router...")
        APP_STATE["model_router"] = get_model_router()
        
        print("[PASS] Enhanced intelligence systems initialized")
        return APP_STATE
        
    except Exception as e:
        print(f"ERROR in initialize_enhanced_system: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

# Main build function - enhanced version
def build_agent_graph():
    """Build the agent graph - now returns enhanced version."""
    return build_enhanced_agent_graph()