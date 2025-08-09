# src/core/agent/self_corrector.py

import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .llm_tracker import TokenTrackingLLM, get_global_tracker

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    REQUIRES_RETRY = "requires_retry"

@dataclass
class SelfCorrectionAnalysis:
    validation_result: ValidationResult
    confidence: float
    issues_found: List[str]
    suggested_corrections: List[str]
    reasoning: str
    should_retry: bool
    retry_instructions: Optional[str] = None

class SelfCorrectingAgent:
    """
    Agent that validates its own work and can trigger corrections.
    Analyzes if the SQL result makes logical sense given the original query.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize LLM for self-correction
        openai_config = self.config.get('openai', {})
        self.llm = TokenTrackingLLM(
            model=openai_config.get('chat_model', 'gpt-4o-mini'),
            temperature=0.1  # Low temperature for consistency
        )
        
        # Register with global tracker
        get_global_tracker().register_llm("self_corrector", self.llm)
        
        # Common validation patterns
        self.common_issues = {
            'empty_result': "The query returned no results",
            'unexpected_columns': "The result columns don't match what was asked",
            'wrong_aggregation': "The aggregation doesn't match the query intent",
            'missing_filters': "Results seem to ignore filtering conditions",
            'incorrect_joins': "Data appears to be from wrong table relationships",
            'scale_mismatch': "Numbers are significantly different than expected"
        }
        
    def validate_result(self, 
                       user_query: str, 
                       sql_query: str, 
                       result_data: Any,
                       context: Optional[Dict] = None) -> SelfCorrectionAnalysis:
        """
        Main validation method that checks if the SQL result makes sense
        for the given user query.
        """
        try:
            # Quick checks first
            quick_issues = self._perform_quick_validation(user_query, sql_query, result_data)
            
            # If quick validation finds serious issues, return immediately
            if len(quick_issues) > 2:
                return SelfCorrectionAnalysis(
                    validation_result=ValidationResult.INVALID,
                    confidence=0.9,
                    issues_found=quick_issues,
                    suggested_corrections=self._generate_corrections(quick_issues),
                    reasoning="Multiple structural issues detected",
                    should_retry=True,
                    retry_instructions="Fix structural issues and regenerate SQL"
                )
            
            # Deep semantic validation using LLM
            semantic_analysis = self._perform_semantic_validation(
                user_query, sql_query, result_data, context
            )
            
            return semantic_analysis
            
        except Exception as e:
            logger.error(f"Self-correction validation failed: {e}")
            return SelfCorrectionAnalysis(
                validation_result=ValidationResult.SUSPICIOUS,
                confidence=0.1,
                issues_found=[f"Validation error: {str(e)}"],
                suggested_corrections=["Manual review recommended"],
                reasoning="Validation process encountered an error",
                should_retry=False
            )
    
    def _perform_quick_validation(self, user_query: str, sql_query: str, result_data: Any) -> List[str]:
        """Perform quick structural validations."""
        issues = []
        
        # Check if result is empty when it shouldn't be
        if isinstance(result_data, pd.DataFrame):
            if len(result_data) == 0 and "count" not in user_query.lower():
                issues.append("empty_result")
        elif isinstance(result_data, str) and "Error:" in result_data:
            issues.append("sql_execution_error")
        
        # Check for common SQL patterns vs query intent
        user_lower = user_query.lower()
        sql_lower = sql_query.lower() if sql_query else ""
        
        # Aggregation checks
        if any(word in user_lower for word in ['total', 'sum', 'average', 'count', 'max', 'min']):
            if not any(func in sql_lower for func in ['sum(', 'avg(', 'count(', 'max(', 'min(']):
                issues.append("missing_aggregation")
        
        # Filtering checks  
        if any(word in user_lower for word in ['where', 'filter', 'only', 'specific']):
            if 'where' not in sql_lower:
                issues.append("missing_filters")
        
        # Sorting checks
        if any(word in user_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst']):
            if 'order by' not in sql_lower or 'limit' not in sql_lower:
                issues.append("missing_sorting_or_limit")
        
        return issues
    
    def _perform_semantic_validation(self, 
                                   user_query: str, 
                                   sql_query: str, 
                                   result_data: Any,
                                   context: Optional[Dict] = None) -> SelfCorrectionAnalysis:
        """Perform deep semantic validation using LLM."""
        
        # Prepare result summary for LLM analysis
        result_summary = self._summarize_result(result_data)
        
        validation_prompt = self._build_validation_prompt(
            user_query, sql_query, result_summary, context
        )
        
        try:
            response = self.llm.invoke(validation_prompt, call_type="self_correction")
            analysis = self._parse_validation_response(response.content)
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return SelfCorrectionAnalysis(
                validation_result=ValidationResult.SUSPICIOUS,
                confidence=0.2,
                issues_found=["LLM validation failed"],
                suggested_corrections=["Manual review needed"],
                reasoning=f"Could not complete semantic validation: {e}",
                should_retry=False
            )
    
    def _summarize_result(self, result_data: Any) -> str:
        """Create a concise summary of the result for LLM analysis."""
        if isinstance(result_data, pd.DataFrame):
            if len(result_data) == 0:
                return "Empty DataFrame (0 rows)"
            
            summary = f"DataFrame with {len(result_data)} rows and {len(result_data.columns)} columns.\n"
            summary += f"Columns: {', '.join(result_data.columns.tolist())}\n"
            
            # Show first few rows (limit for token efficiency)
            if len(result_data) <= 5:
                summary += "All rows:\n" + result_data.to_string(max_cols=6)
            else:
                summary += "Sample rows:\n" + result_data.head(3).to_string(max_cols=6)
                summary += f"\n... ({len(result_data)-3} more rows)"
            
            # Basic statistics for numeric columns
            numeric_cols = result_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary += f"\nNumeric column statistics:\n"
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    stats = result_data[col].describe()
                    summary += f"{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}\n"
        
        elif isinstance(result_data, str):
            return f"String result: {result_data[:500]}..."  # Limit length
        
        else:
            return f"Result type: {type(result_data)}, Value: {str(result_data)[:200]}..."
        
        return summary
    
    def _build_validation_prompt(self, 
                               user_query: str, 
                               sql_query: str, 
                               result_summary: str,
                               context: Optional[Dict] = None) -> str:
        """Build the validation prompt for LLM analysis."""
        
        context_info = ""
        if context:
            context_info = f"\nDatabase Context: {json.dumps(context, indent=2)}"
        
        return f"""
You are a data analyst validator. Your job is to check if a SQL query result makes logical sense given the original user question.

ORIGINAL USER QUERY: {user_query}

GENERATED SQL QUERY: {sql_query}

RESULT SUMMARY: {result_summary}
{context_info}

Please analyze whether the result makes sense for the user's question. Consider:
1. Does the result structure match what was asked?
2. Do the data values seem reasonable?
3. Are the columns/aggregations correct?
4. Does the result size make sense?
5. Are there any obvious logical inconsistencies?

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of specific issues found"],
    "reasoning": "detailed explanation of your analysis",
    "should_retry": true/false,
    "retry_instructions": "specific instructions for fixing the query if retry needed"
}}

Be thorough but concise. Focus on logical correctness rather than minor formatting issues.
"""
    
    def _parse_validation_response(self, response_content: str) -> SelfCorrectionAnalysis:
        """Parse the LLM validation response."""
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_str = response_content[json_start:json_end]
            else:
                # Try to find JSON directly
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1
                json_str = response_content[json_start:json_end]
            
            parsed = json.loads(json_str)
            
            # Determine validation result
            if parsed.get("is_valid", False):
                if parsed.get("confidence", 0) > 0.8:
                    result = ValidationResult.VALID
                else:
                    result = ValidationResult.SUSPICIOUS
            else:
                result = ValidationResult.INVALID
                
            return SelfCorrectionAnalysis(
                validation_result=result,
                confidence=parsed.get("confidence", 0.5),
                issues_found=parsed.get("issues", []),
                suggested_corrections=self._generate_corrections(parsed.get("issues", [])),
                reasoning=parsed.get("reasoning", "LLM validation analysis"),
                should_retry=parsed.get("should_retry", False),
                retry_instructions=parsed.get("retry_instructions")
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {e}")
            # Fallback to simple text analysis
            is_valid = "valid" in response_content.lower() and "not valid" not in response_content.lower()
            confidence = 0.6 if is_valid else 0.4
            
            return SelfCorrectionAnalysis(
                validation_result=ValidationResult.VALID if is_valid else ValidationResult.SUSPICIOUS,
                confidence=confidence,
                issues_found=["Could not fully parse validation response"],
                suggested_corrections=["Manual review recommended"],
                reasoning="Partial validation based on text analysis",
                should_retry=not is_valid
            )
    
    def _generate_corrections(self, issues: List[str]) -> List[str]:
        """Generate correction suggestions based on identified issues."""
        corrections = []
        
        for issue in issues:
            if "empty_result" in issue:
                corrections.append("Check if filters are too restrictive or table names are correct")
            elif "missing_aggregation" in issue:
                corrections.append("Add appropriate aggregation functions (SUM, COUNT, AVG, etc.)")
            elif "missing_filters" in issue:
                corrections.append("Add WHERE clause to filter results as requested")
            elif "missing_sorting" in issue:
                corrections.append("Add ORDER BY clause and LIMIT for top/bottom results")
            elif "sql_execution_error" in issue:
                corrections.append("Fix SQL syntax or schema issues")
            else:
                corrections.append(f"Address issue: {issue}")
        
        if not corrections:
            corrections.append("Review query logic and ensure it addresses the user's intent")
        
        return corrections


# Global instance
_self_corrector = None

def get_self_corrector(config: Dict = None) -> SelfCorrectingAgent:
    """Get or create global self-corrector instance."""
    global _self_corrector
    if _self_corrector is None:
        _self_corrector = SelfCorrectingAgent(config=config)
    return _self_corrector

def reset_self_corrector(config: Dict = None) -> SelfCorrectingAgent:
    """Reset self-corrector instance."""
    global _self_corrector
    _self_corrector = SelfCorrectingAgent(config=config)
    return _self_corrector
