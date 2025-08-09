# src/core/agent/query_classifier.py

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class QueryIntent(Enum):
    LOOKUP = "lookup"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    TREND = "trend"
    RELATIONSHIP = "relationship"

class QueryDomain(Enum):
    GENERAL = "general"
    FINANCIAL = "financial"
    SALES = "sales"
    INVENTORY = "inventory"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"

class QueryAmbiguity(Enum):
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    REQUIRES_CLARIFICATION = "requires_clarification"

@dataclass
class AmbiguityAnalysis:
    ambiguity_level: QueryAmbiguity
    ambiguous_terms: List[str]
    clarification_needed: List[str]
    confidence: float
    suggested_clarifications: List[Dict[str, str]]

@dataclass
class QueryClassification:
    complexity: QueryComplexity
    intent: QueryIntent
    domain: QueryDomain
    confidence: float
    reasoning: str
    estimated_cost: int  # Token estimate
    recommended_model: str
    ambiguity_analysis: Optional[AmbiguityAnalysis] = None

class IntelligentQueryClassifier:
    """Advanced query classification system for optimal model routing."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.classification_config = self.config.get('query_classification', {})
        
        # Complexity indicators
        self.simple_patterns = [
            r"^(what is|show me|list|display|get)\s+\w+",
            r"^count\s+",
            r"^find\s+\w+\s+with\s+\w+\s*=",
            r"^(how many|total)\s+",
        ]
        
        self.complex_patterns = [
            r"(correlation|relationship|trend|pattern)",
            r"(compare|contrast|vs|versus)",
            r"(analyze|analysis|insight|deep dive)",
            r"(predict|forecast|projection)",
            r"(multiple|several|various)\s+(tables|datasets)",
            r"(group\s+by.*order\s+by|having|window\s+function)",
        ]
        
        # Domain keywords
        self.domain_keywords = {
            QueryDomain.FINANCIAL: ["revenue", "profit", "cost", "expense", "budget", "financial", "accounting", "balance"],
            QueryDomain.SALES: ["sales", "customer", "order", "purchase", "transaction", "deal", "quota"],
            QueryDomain.INVENTORY: ["inventory", "stock", "product", "item", "warehouse", "supply"],
            QueryDomain.CUSTOMER: ["customer", "client", "user", "demographics", "behavior", "segment"],
            QueryDomain.OPERATIONAL: ["operation", "process", "efficiency", "performance", "productivity"],
        }
        
        # Intent keywords
        self.intent_keywords = {
            QueryIntent.LOOKUP: ["show", "display", "get", "find", "list", "what is"],
            QueryIntent.AGGREGATION: ["total", "sum", "average", "count", "max", "min", "aggregate"],
            QueryIntent.COMPARISON: ["compare", "vs", "versus", "difference", "between", "contrast"],
            QueryIntent.ANALYSIS: ["analyze", "analysis", "insight", "pattern", "why", "how", "explain"],
            QueryIntent.TREND: ["trend", "over time", "growth", "change", "evolution", "timeline"],
            QueryIntent.RELATIONSHIP: ["relationship", "correlation", "connection", "impact", "effect"],
        }
        
        # Ambiguity patterns for HITL
        self.ambiguous_patterns = {
            'person_names': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # John Smith, Mary Johnson
            'generic_names': r'\b(john|mary|smith|johnson|mike|sarah|david|lisa)\b',
            'vague_references': r'\b(this|that|these|those|it|them)\b',
            'incomplete_dates': r'\b(last|this|next)\s+(week|month|quarter|year)\b',
            'ambiguous_identifiers': r'\b(product|customer|order|item)\s+\d+\b',
            'multiple_meanings': r'\b(sales|revenue|profit|margin|cost)\b(?!\s+(by|for|of|in|from))',
        }

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def classify_query(self, query: str, context: Optional[Dict] = None) -> QueryClassification:
        """Main classification method with comprehensive analysis."""
        query_lower = query.lower().strip()
        
        # Determine complexity
        complexity, complexity_score = self._assess_complexity(query_lower)
        
        # Determine intent
        intent, intent_confidence = self._classify_intent(query_lower)
        
        # Determine domain
        domain, domain_confidence = self._classify_domain(query_lower, context)
        
        # Analyze ambiguity for HITL
        ambiguity_analysis = self._analyze_ambiguity(query, query_lower, context)
        
        # Calculate overall confidence
        overall_confidence = (complexity_score + intent_confidence + domain_confidence) / 3
        
        # Estimate token cost
        estimated_cost = self._estimate_token_cost(query, complexity, intent)
        
        # Recommend model
        recommended_model = self._recommend_model(complexity, intent, estimated_cost)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(complexity, intent, domain, complexity_score)
        
        return QueryClassification(
            complexity=complexity,
            intent=intent,
            domain=domain,
            confidence=overall_confidence,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            recommended_model=recommended_model,
            ambiguity_analysis=ambiguity_analysis
        )

    def _assess_complexity(self, query: str) -> Tuple[QueryComplexity, float]:
        """Assess query complexity based on multiple factors."""
        complexity_score = 0.0
        factors = []
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 0.3
            factors.append("long_query")
        elif word_count < 5:
            complexity_score -= 0.2
            factors.append("short_query")
        
        # Pattern matching
        simple_matches = sum(1 for pattern in self.simple_patterns if re.search(pattern, query, re.IGNORECASE))
        complex_matches = sum(1 for pattern in self.complex_patterns if re.search(pattern, query, re.IGNORECASE))
        
        if simple_matches > complex_matches:
            complexity_score -= 0.3
            factors.append("simple_patterns")
        elif complex_matches > simple_matches:
            complexity_score += 0.4
            factors.append("complex_patterns")
        
        # SQL complexity indicators
        sql_indicators = ["join", "group by", "having", "subquery", "case when", "window", "recursive"]
        sql_complexity = sum(1 for indicator in sql_indicators if indicator in query)
        complexity_score += sql_complexity * 0.1
        
        # Multiple table references
        if len(re.findall(r'\b\w+\.\w+\b', query)) > 2:
            complexity_score += 0.2
            factors.append("multiple_tables")
        
        # Statistical/analytical terms
        analytical_terms = ["correlation", "regression", "variance", "deviation", "percentile", "distribution"]
        if any(term in query for term in analytical_terms):
            complexity_score += 0.3
            factors.append("statistical_analysis")
        
        # Normalize score
        complexity_score = max(0.0, min(1.0, complexity_score + 0.5))  # Baseline 0.5
        
        # Map to complexity level
        if complexity_score < 0.4:
            return QueryComplexity.SIMPLE, complexity_score
        elif complexity_score < 0.7:
            return QueryComplexity.MODERATE, complexity_score
        else:
            return QueryComplexity.COMPLEX, complexity_score

    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify the intent of the query."""
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.LOOKUP, 0.5  # Default
        
        # Get the intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_intent[1] / 3)  # Normalize
        
        return best_intent[0], confidence

    def _classify_domain(self, query: str, context: Optional[Dict] = None) -> Tuple[QueryDomain, float]:
        """Classify the business domain of the query."""
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                domain_scores[domain] = score
        
        # Use context if available (table names, column names)
        if context and "table_info" in context:
            table_names = context["table_info"].keys()
            for domain, keywords in self.domain_keywords.items():
                context_score = sum(1 for table in table_names 
                                  for keyword in keywords 
                                  if keyword in table.lower())
                domain_scores[domain] = domain_scores.get(domain, 0) + context_score * 0.5
        
        if not domain_scores:
            return QueryDomain.GENERAL, 0.5
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_domain[1] / 3)
        
        return best_domain[0], confidence

    def _estimate_token_cost(self, query: str, complexity: QueryComplexity, intent: QueryIntent) -> int:
        """Estimate token cost for the query processing."""
        base_cost = len(query.split()) * 1.3  # Base query tokens
        
        # Context retrieval cost
        context_cost = {
            QueryComplexity.SIMPLE: 500,
            QueryComplexity.MODERATE: 1000,
            QueryComplexity.COMPLEX: 2000
        }[complexity]
        
        # Processing cost based on intent
        processing_cost = {
            QueryIntent.LOOKUP: 200,
            QueryIntent.AGGREGATION: 400,
            QueryIntent.COMPARISON: 600,
            QueryIntent.ANALYSIS: 1000,
            QueryIntent.TREND: 800,
            QueryIntent.RELATIONSHIP: 1200
        }[intent]
        
        # Response generation cost
        response_cost = 300
        
        return int(base_cost + context_cost + processing_cost + response_cost)

    def _recommend_model(self, complexity: QueryComplexity, intent: QueryIntent, estimated_cost: int) -> str:
        """Recommend the optimal model based on classification."""
        
        # Force complex queries to use gpt-4o
        if complexity == QueryComplexity.COMPLEX:
            return "gpt-4o"
        
        # Analysis and relationship queries benefit from gpt-4o
        if intent in [QueryIntent.ANALYSIS, QueryIntent.RELATIONSHIP]:
            return "gpt-4o"
        
        # High-cost queries should use the better model
        if estimated_cost > 1500:
            return "gpt-4o"
        
        # Default to cost-effective model
        return "gpt-4o-mini"

    def _generate_reasoning(self, complexity: QueryComplexity, intent: QueryIntent, 
                          domain: QueryDomain, complexity_score: float) -> str:
        """Generate human-readable reasoning for the classification."""
        reasons = []
        
        reasons.append(f"Complexity: {complexity.value} (score: {complexity_score:.2f})")
        reasons.append(f"Intent: {intent.value}")
        reasons.append(f"Domain: {domain.value}")
        
        if complexity == QueryComplexity.COMPLEX:
            reasons.append("Requires advanced reasoning capabilities")
        elif complexity == QueryComplexity.SIMPLE:
            reasons.append("Straightforward data retrieval")
        
        return " | ".join(reasons)

    def _analyze_ambiguity(self, original_query: str, query_lower: str, context: Optional[Dict] = None) -> AmbiguityAnalysis:
        """Analyze query for ambiguous terms and need for clarification."""
        ambiguous_terms = []
        clarification_needed = []
        suggested_clarifications = []
        
        # Check for person names that could be ambiguous
        person_matches = re.findall(self.ambiguous_patterns['person_names'], original_query)
        generic_matches = re.findall(self.ambiguous_patterns['generic_names'], query_lower)
        
        if person_matches or generic_matches:
            ambiguous_terms.extend(person_matches + generic_matches)
            clarification_needed.append("person_identity")
            suggested_clarifications.append({
                "type": "person_clarification",
                "message": f"I found references to people ({', '.join(set(person_matches + generic_matches))}). Could you specify which person you mean?",
                "options": ["Search by name", "Search by employee ID", "Search by department", "Show all matching names"]
            })
        
        # Check for vague references
        vague_matches = re.findall(self.ambiguous_patterns['vague_references'], query_lower)
        if vague_matches:
            ambiguous_terms.extend(vague_matches)
            clarification_needed.append("vague_reference")
            suggested_clarifications.append({
                "type": "reference_clarification", 
                "message": f"Your query contains vague references ({', '.join(set(vague_matches))}). Could you be more specific?",
                "options": ["Specify the item", "Provide more context", "Use specific names/IDs"]
            })
        
        # Check for incomplete date references
        date_matches = re.findall(self.ambiguous_patterns['incomplete_dates'], query_lower)
        if date_matches:
            ambiguous_terms.extend([' '.join(match) for match in date_matches])
            clarification_needed.append("time_period")
            suggested_clarifications.append({
                "type": "date_clarification",
                "message": f"I found time references ({', '.join([' '.join(match) for match in date_matches])}). Could you specify the exact dates?",
                "options": ["Specify exact date range", "Use current period", "Select from calendar"]
            })
        
        # Check for ambiguous identifiers
        id_matches = re.findall(self.ambiguous_patterns['ambiguous_identifiers'], query_lower)
        if id_matches:
            ambiguous_terms.extend(id_matches)
            clarification_needed.append("identifier_type")
            suggested_clarifications.append({
                "type": "identifier_clarification",
                "message": f"I found identifiers ({', '.join(set(id_matches))}). Could you clarify what type of ID this is?",
                "options": ["Product ID", "Customer ID", "Order ID", "Reference number"]
            })
        
        # Check for terms with multiple meanings
        multi_meaning_matches = re.findall(self.ambiguous_patterns['multiple_meanings'], query_lower)
        if multi_meaning_matches and not any(re.search(rf'{term}\s+(by|for|of|in|from)', query_lower) for term in multi_meaning_matches):
            ambiguous_terms.extend(multi_meaning_matches)
            clarification_needed.append("metric_specification")
            suggested_clarifications.append({
                "type": "metric_clarification",
                "message": f"The terms ({', '.join(set(multi_meaning_matches))}) could have multiple meanings. Could you be more specific?",
                "options": ["Specify the metric type", "Provide calculation method", "Give example of expected result"]
            })
        
        # Determine ambiguity level
        ambiguity_confidence = 1.0
        if len(ambiguous_terms) == 0:
            ambiguity_level = QueryAmbiguity.CLEAR
        elif len(clarification_needed) >= 2 or "person_identity" in clarification_needed:
            ambiguity_level = QueryAmbiguity.REQUIRES_CLARIFICATION
            ambiguity_confidence = 0.3
        else:
            ambiguity_level = QueryAmbiguity.AMBIGUOUS
            ambiguity_confidence = 0.6
        
        return AmbiguityAnalysis(
            ambiguity_level=ambiguity_level,
            ambiguous_terms=list(set(ambiguous_terms)),
            clarification_needed=clarification_needed,
            confidence=ambiguity_confidence,
            suggested_clarifications=suggested_clarifications
        )

class ModelRouter:
    """Intelligent model routing system."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model_config = self.config.get('openai', {})
        
        # Model capabilities and costs
        self.models = {
            "gpt-4o-mini": {
                "input_cost": 0.15,   # per 1M tokens
                "output_cost": 0.6,   # per 1M tokens
                "max_tokens": 128000,
                "best_for": ["simple", "lookup", "aggregation"]
            },
            "gpt-4o": {
                "input_cost": 5.0,    # per 1M tokens
                "output_cost": 15.0,  # per 1M tokens
                "max_tokens": 128000,
                "best_for": ["complex", "analysis", "reasoning"]
            }
        }

    def route_query(self, classification: QueryClassification) -> Dict:
        """Route query to optimal model with cost consideration."""
        recommended_model = classification.recommended_model
        
        # Override based on cost constraints if configured
        cost_threshold = self.config.get('cost_optimization', {}).get('max_tokens_per_query', 3000)
        
        if classification.estimated_cost > cost_threshold and recommended_model == "gpt-4o":
            # Check if we can downgrade without significant quality loss
            if classification.complexity != QueryComplexity.COMPLEX:
                recommended_model = "gpt-4o-mini"
        
        model_info = self.models[recommended_model]
        
        return {
            "model": recommended_model,
            "temperature": self.model_config.get('temperature', 0),
            "max_tokens": min(4000, model_info["max_tokens"]),
            "estimated_cost": self._calculate_cost(classification.estimated_cost, recommended_model),
            "reasoning": f"Selected {recommended_model} for {classification.complexity.value} {classification.intent.value} query"
        }

    def _calculate_cost(self, estimated_tokens: int, model: str) -> float:
        """Calculate estimated cost in USD."""
        model_info = self.models[model]
        input_cost = (estimated_tokens * 0.7 / 1000000) * model_info["input_cost"]  # 70% input
        output_cost = (estimated_tokens * 0.3 / 1000000) * model_info["output_cost"] # 30% output
        return input_cost + output_cost

# Global instances
_classifier = None
_router = None

def get_query_classifier() -> IntelligentQueryClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = IntelligentQueryClassifier()
    return _classifier

def get_model_router() -> ModelRouter:
    """Get or create global router instance."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router