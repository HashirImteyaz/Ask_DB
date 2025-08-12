# src/core/agent/query_analyzer.py

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..data_processing.multi_retrieval_system import RetrievalType


class QueryFocus(Enum):
    """Types of information focus in user queries."""
    COLUMN_DETAILS = "column_details"
    TABLE_STRUCTURE = "table_structure"
    DATA_VALUES = "data_values"
    RELATIONSHIPS = "relationships"
    SCHEMA_OVERVIEW = "schema_overview"
    MIXED = "mixed"


@dataclass
class QueryAnalysis:
    """Analysis of user query for retrieval strategy."""
    primary_focus: QueryFocus
    secondary_focus: Optional[QueryFocus]
    mentioned_tables: List[str]
    mentioned_columns: List[str]
    retrieval_strategy: List[RetrievalType]
    confidence: float
    reasoning: str


class IntelligentQueryAnalyzer:
    """Analyzes user queries to determine optimal retrieval strategy."""
    
    def __init__(self, schema_data: Optional[Dict] = None):
        self.schema_data = schema_data or {}
        self.table_names = set()
        self.column_names = set()
        
        # Extract table and column names from schema
        if schema_data:
            self.table_names = set(schema_data.get("table_descriptions", {}).keys())
            column_descriptions = schema_data.get("column_descriptions", {})
            for table_cols in column_descriptions.values():
                self.column_names.update(table_cols.keys())
        
        # Define query patterns for different focuses
        self.focus_patterns = {
            QueryFocus.COLUMN_DETAILS: [
                r'\b(column|field|attribute|property)\b',
                r'\b(data type|type of|what type)\b',
                r'\b(description|meaning|purpose) of (\w+)',
                r'\b(explain|describe) (\w+) (column|field)',
                r'\bwhat (is|does) (\w+) (mean|represent)',
                r'\b(column|field) (\w+)',
                r'\bdetails? (about|of) (\w+) (column|field)',
            ],
            
            QueryFocus.TABLE_STRUCTURE: [
                r'\b(table|entity|schema) structure\b',
                r'\bwhat (tables?|entities?) (do|are|exist)',
                r'\b(describe|explain) (the )?table\b',
                r'\btable (\w+) (structure|schema|definition)',
                r'\bwhich tables?\b',
                r'\btables? (available|in|for)',
                r'\bschema (information|details|overview)',
            ],
            
            QueryFocus.DATA_VALUES: [
                r'\b(values?|examples?|samples?) (in|of|for)\b',
                r'\bwhat (values?|data) (are|is) (in|stored)',
                r'\b(possible|valid|available) values?\b',
                r'\b(range|min|max|minimum|maximum) (of|for|in)',
                r'\bdistinct values?\b',
                r'\bunique values?\b',
                r'\bdata (in|inside|within)',
            ],
            
            QueryFocus.RELATIONSHIPS: [
                r'\b(relationship|relation|connection) between\b',
                r'\bhow (are|do) (\w+) (and|&) (\w+) (related|connected)',
                r'\b(join|link|connect) (\w+) (with|to|and)',
                r'\bforeign key\b',
                r'\bprimary key\b',
                r'\btables? (related|connected|linked) to\b',
                r'\brelations?hip\b',
            ],
            
            QueryFocus.SCHEMA_OVERVIEW: [
                r'\b(overview|summary) (of|for) (the )?(database|schema|tables?)',
                r'\bwhat (is|does) (the )?(database|schema) (contain|have)',
                r'\b(all|list) (tables?|entities?)\b',
                r'\bdatabase (structure|schema|information)',
                r'\bgeneral (information|overview)',
                r'\bwhat data (do|is) (you|we|I) have',
            ]
        }
        
        # Patterns to identify specific table/column mentions
        self.table_mention_patterns = [
            r'\btable (\w+)\b',
            r'\bin (\w+) table\b',
            r'\bfrom (\w+)\b',
            r'\b(\w+) (table|entity)\b',
        ]
        
        self.column_mention_patterns = [
            r'\bcolumn (\w+)\b',
            r'\bfield (\w+)\b',
            r'\b(\w+) (column|field|attribute)\b',
            r'\bvalue of (\w+)\b',
            r'\b(\w+) values?\b',
        ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine retrieval strategy."""
        query_lower = query.lower()
        
        # Identify mentioned tables and columns
        mentioned_tables = self._extract_mentioned_tables(query_lower)
        mentioned_columns = self._extract_mentioned_columns(query_lower)
        
        # Determine query focus
        primary_focus, secondary_focus, focus_confidence = self._determine_query_focus(query_lower)
        
        # Determine retrieval strategy
        retrieval_strategy = self._determine_retrieval_strategy(
            primary_focus, secondary_focus, mentioned_tables, mentioned_columns
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_focus, secondary_focus, mentioned_tables, mentioned_columns, retrieval_strategy
        )
        
        return QueryAnalysis(
            primary_focus=primary_focus,
            secondary_focus=secondary_focus,
            mentioned_tables=mentioned_tables,
            mentioned_columns=mentioned_columns,
            retrieval_strategy=retrieval_strategy,
            confidence=focus_confidence,
            reasoning=reasoning
        )
    
    def _extract_mentioned_tables(self, query_lower: str) -> List[str]:
        """Extract explicitly mentioned table names from query."""
        mentioned = set()
        
        # Use patterns to find table mentions
        for pattern in self.table_mention_patterns:
            matches = re.findall(pattern, query_lower)
            mentioned.update(matches)
        
        # Check against known table names
        valid_tables = []
        for table in mentioned:
            # Exact match
            if table in self.table_names:
                valid_tables.append(table)
            else:
                # Fuzzy match (partial)
                for known_table in self.table_names:
                    if table in known_table.lower() or known_table.lower() in table:
                        valid_tables.append(known_table)
                        break
        
        # Also check for table names mentioned without explicit keywords
        for table_name in self.table_names:
            table_lower = table_name.lower()
            if table_lower in query_lower and table_name not in valid_tables:
                valid_tables.append(table_name)
        
        return valid_tables
    
    def _extract_mentioned_columns(self, query_lower: str) -> List[str]:
        """Extract explicitly mentioned column names from query."""
        mentioned = set()
        
        # Use patterns to find column mentions
        for pattern in self.column_mention_patterns:
            matches = re.findall(pattern, query_lower)
            mentioned.update(matches)
        
        # Check against known column names
        valid_columns = []
        for column in mentioned:
            # Exact match
            if column in self.column_names:
                valid_columns.append(column)
            else:
                # Fuzzy match (partial)
                for known_column in self.column_names:
                    if column in known_column.lower() or known_column.lower() in column:
                        valid_columns.append(known_column)
                        break
        
        # Also check for column names mentioned without explicit keywords
        for column_name in self.column_names:
            column_lower = column_name.lower()
            if column_lower in query_lower and column_name not in valid_columns:
                valid_columns.append(column_name)
        
        return valid_columns
    
    def _determine_query_focus(self, query_lower: str) -> Tuple[QueryFocus, Optional[QueryFocus], float]:
        """Determine the primary and secondary focus of the query."""
        focus_scores = {}
        
        # Score each focus type based on pattern matches
        for focus_type, patterns in self.focus_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            
            if score > 0:
                focus_scores[focus_type] = score
        
        # Handle case where no specific patterns match
        if not focus_scores:
            # Default heuristics
            if any(word in query_lower for word in ['column', 'field', 'attribute', 'data type']):
                focus_scores[QueryFocus.COLUMN_DETAILS] = 1
            elif any(word in query_lower for word in ['table', 'schema', 'structure']):
                focus_scores[QueryFocus.TABLE_STRUCTURE] = 1
            elif any(word in query_lower for word in ['value', 'example', 'sample']):
                focus_scores[QueryFocus.DATA_VALUES] = 1
            else:
                focus_scores[QueryFocus.MIXED] = 1
        
        # Sort by score
        sorted_focuses = sorted(focus_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_focus = sorted_focuses[0][0]
        secondary_focus = sorted_focuses[1][0] if len(sorted_focuses) > 1 and sorted_focuses[1][1] > 0 else None
        
        # Calculate confidence
        total_score = sum(focus_scores.values())
        primary_score = sorted_focuses[0][1]
        confidence = min(0.95, primary_score / max(1, total_score))
        
        # If multiple focuses with similar scores, it's mixed
        if len(sorted_focuses) > 1 and sorted_focuses[1][1] / primary_score > 0.7:
            primary_focus = QueryFocus.MIXED
            confidence = 0.6
        
        return primary_focus, secondary_focus, confidence
    
    def _determine_retrieval_strategy(self, primary_focus: QueryFocus, secondary_focus: Optional[QueryFocus],
                                    mentioned_tables: List[str], mentioned_columns: List[str]) -> List[RetrievalType]:
        """Determine which retrieval systems to use."""
        strategy = []
        
        # Strategy based on primary focus
        if primary_focus == QueryFocus.COLUMN_DETAILS:
            strategy.append(RetrievalType.COLUMN_DESCRIPTIONS)
            if mentioned_tables:
                strategy.append(RetrievalType.TABLE_DESCRIPTIONS)
        
        elif primary_focus == QueryFocus.TABLE_STRUCTURE:
            strategy.append(RetrievalType.TABLE_DESCRIPTIONS)
            if mentioned_columns:
                strategy.append(RetrievalType.COLUMN_DESCRIPTIONS)
        
        elif primary_focus == QueryFocus.DATA_VALUES:
            strategy.append(RetrievalType.COLUMN_DESCRIPTIONS)  # Column descriptions include value examples
        
        elif primary_focus == QueryFocus.RELATIONSHIPS:
            strategy.extend([RetrievalType.TABLE_DESCRIPTIONS, RetrievalType.COLUMN_DESCRIPTIONS])
        
        elif primary_focus == QueryFocus.SCHEMA_OVERVIEW:
            strategy.append(RetrievalType.HYBRID)
        
        elif primary_focus == QueryFocus.MIXED:
            strategy.append(RetrievalType.HYBRID)
        
        # Add secondary focus considerations
        if secondary_focus and secondary_focus != primary_focus:
            if secondary_focus == QueryFocus.COLUMN_DETAILS and RetrievalType.COLUMN_DESCRIPTIONS not in strategy:
                strategy.append(RetrievalType.COLUMN_DESCRIPTIONS)
            elif secondary_focus == QueryFocus.TABLE_STRUCTURE and RetrievalType.TABLE_DESCRIPTIONS not in strategy:
                strategy.append(RetrievalType.TABLE_DESCRIPTIONS)
        
        # If specific tables or columns mentioned, ensure appropriate retrievers
        if mentioned_tables and RetrievalType.TABLE_DESCRIPTIONS not in strategy:
            strategy.append(RetrievalType.TABLE_DESCRIPTIONS)
        
        if mentioned_columns and RetrievalType.COLUMN_DESCRIPTIONS not in strategy:
            strategy.append(RetrievalType.COLUMN_DESCRIPTIONS)
        
        # Fallback
        if not strategy:
            strategy.append(RetrievalType.HYBRID)
        
        return strategy
    
    def _generate_reasoning(self, primary_focus: QueryFocus, secondary_focus: Optional[QueryFocus],
                          mentioned_tables: List[str], mentioned_columns: List[str],
                          retrieval_strategy: List[RetrievalType]) -> str:
        """Generate human-readable reasoning for the analysis."""
        reasons = []
        
        # Primary focus reasoning
        reasons.append(f"Primary focus: {primary_focus.value.replace('_', ' ')}")
        
        if secondary_focus:
            reasons.append(f"Secondary focus: {secondary_focus.value.replace('_', ' ')}")
        
        # Mentions reasoning
        if mentioned_tables:
            reasons.append(f"Mentions tables: {', '.join(mentioned_tables)}")
        
        if mentioned_columns:
            reasons.append(f"Mentions columns: {', '.join(mentioned_columns)}")
        
        # Strategy reasoning
        strategy_desc = []
        for ret_type in retrieval_strategy:
            if ret_type == RetrievalType.COLUMN_DESCRIPTIONS:
                strategy_desc.append("column-focused retrieval")
            elif ret_type == RetrievalType.TABLE_DESCRIPTIONS:
                strategy_desc.append("table-focused retrieval")
            elif ret_type == RetrievalType.HYBRID:
                strategy_desc.append("hybrid retrieval")
        
        if strategy_desc:
            reasons.append(f"Strategy: {', '.join(strategy_desc)}")
        
        return " | ".join(reasons)


def analyze_query_for_retrieval(query: str, schema_data: Optional[Dict] = None) -> QueryAnalysis:
    """Convenience function to analyze a query for retrieval strategy."""
    analyzer = IntelligentQueryAnalyzer(schema_data)
    return analyzer.analyze_query(query)


def get_retrieval_recommendation(query: str, schema_data: Optional[Dict] = None) -> Dict:
    """Get retrieval recommendation in a simple format."""
    analysis = analyze_query_for_retrieval(query, schema_data)
    
    return {
        "retrieval_types": [rt.value for rt in analysis.retrieval_strategy],
        "focus_areas": [analysis.primary_focus.value] + ([analysis.secondary_focus.value] if analysis.secondary_focus else []),
        "mentioned_entities": {
            "tables": analysis.mentioned_tables,
            "columns": analysis.mentioned_columns
        },
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning
    }
