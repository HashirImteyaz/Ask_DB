
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any, List
from pandas import DataFrame
from .query_classifier import QueryClassification, AmbiguityAnalysis

class AgentState(BaseModel):
    """Enhanced agent state with intelligence classification and optimization features."""
    
    # Core query information
    user_query: str = Field(..., description="The initial query from the user.")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="The history of the conversation.")
    
    # Enhanced query processing
    query_classification: Optional[QueryClassification] = Field(None, description="Intelligent query classification results")
    complexity_assessment: Optional[Dict[str, Any]] = Field(None, description="Results from query complexity assessment")
    estimated_cost: Optional[int] = Field(None, description="Estimated token cost for processing")
    recommended_model: Optional[str] = Field(None, description="Recommended model for optimal processing")
    confidence_score: Optional[float] = Field(None, description="Confidence score for the response")
    # Human-in-the-Loop (HITL) fields
    requires_clarification: bool = False
    clarification_type: Optional[str] = None
    clarification_options: List[Dict[str, Any]] = Field(default_factory=list, description="Options for user clarification")
    awaiting_user_input: bool = False
    clarification_context: Optional[Dict[str, Any]] = None
    
    # Self-correction fields
    self_correction_needed: bool = False
    correction_analysis: Optional[Dict[str, Any]] = None
    correction_attempt_count: int = 0
    max_correction_attempts: int = 2
    
    # Query refinement
    clarified_query: Optional[str] = None
    original_intent: Optional[str] = Field(None, description="Preserved original user intent")
    
    # Clarification fields
    is_safe: bool = True
    needs_clarification: bool = False
    is_greeting: bool = False
    wants_graph: bool = False
    assistant_response: Optional[str] = None
    clarification_question: Optional[str] = None

    # Enhanced RAG context
    rag_context: Optional[str] = None
    context_token_count: Optional[int] = Field(None, description="Number of tokens used for context")
    context_sources: List[str] = Field(default_factory=list, description="Sources of retrieved context")
    
    # SQL workflow with enhanced tracking and optimization
    sql_query: Optional[str] = None
    sql_result: Optional[Union[str, DataFrame]] = None
    sql_error: Optional[str] = None
    retry_count: int = 0
    sql_optimization_applied: bool = Field(False, description="Whether SQL optimization was applied")
    execution_time: Optional[float] = Field(None, description="SQL execution time in seconds")
    schema_introspection_applied: Optional[bool] = Field(None, description="Whether schema introspection was applied during SQL generation")
    
    # SQL Optimization fields
    optimization_analysis: Optional[Dict[str, Any]] = Field(None, description="SQL optimization analysis results")
    optimization_recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Optimization recommendations")
    original_sql_query: Optional[str] = Field(None, description="Original SQL before optimization")
    optimization_confidence: Optional[float] = Field(None, description="Confidence in optimization suggestions")
    optimization_improvement_percent: Optional[float] = Field(None, description="Estimated performance improvement percentage")
    
    # Memory and context management
    context_history: List[Dict[str, str]] = Field(default_factory=list, description="Relevant conversation slices for context")
    similar_query_context: Optional[Dict[str, Any]] = Field(None, description="Context from similar previous queries")
    from_history: bool = False
    
    # Response generation
    final_answer: Optional[str] = None
    graph_b64: Optional[str] = None
    response_token_count: Optional[int] = Field(None, description="Number of tokens in the response")
    
    # Enhanced features
    suggested_queries: Optional[List[Dict[str, str]]] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance tracking metrics")
    
    # Cost optimization tracking  
    total_cost_estimate: Optional[float] = Field(None, description="Total estimated cost in USD")
    cost_breakdown: Dict[str, float] = Field(default_factory=dict, description="Breakdown of costs by component")
    
    # Quality assurance
    response_quality_score: Optional[float] = Field(None, description="Quality score of the generated response")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Results from response validation")
    
    # Business intelligence
    detected_entities: List[str] = Field(default_factory=list, description="Detected business entities in the query")
    business_context_used: Optional[str] = Field(None, description="Business rules and context applied")
    
    class Config:
        arbitrary_types_allowed = True
        
    def add_performance_metric(self, metric_name: str, value: Any) -> None:
        """Add a performance metric to tracking."""
        self.performance_metrics[metric_name] = value
    
    def calculate_total_tokens(self) -> int:
        """Calculate total tokens used in processing."""
        total = 0
        if self.context_token_count:
            total += self.context_token_count
        if self.response_token_count:
            total += self.response_token_count
        if self.user_query:
            total += len(self.user_query.split()) * 1.3  # Rough estimate
        return int(total)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing metrics."""
        return {
            "query_complexity": self.query_classification.complexity.value if self.query_classification else "unknown",
            "model_used": self.recommended_model,
            "total_tokens": self.calculate_total_tokens(),
            "sql_retries": self.retry_count,
            "used_history": self.from_history,
            "graph_generated": self.graph_b64 is not None,
            "confidence": self.confidence_score,
            "cost_estimate": self.total_cost_estimate
        }