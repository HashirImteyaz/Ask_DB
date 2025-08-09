"""
Advanced SQL Validation and Optimization System
This module provides comprehensive SQL validation, optimization, and performance analysis
"""

import re
import json
import logging
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import inspect, text, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

try:
    import sqlparse
    from sqlparse import sql, tokens
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False
    sqlparse = None

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """SQL validation results."""
    VALID = "valid"
    INVALID_SYNTAX = "invalid_syntax"
    INVALID_SCHEMA = "invalid_schema"
    PERFORMANCE_WARNING = "performance_warning"
    SECURITY_RISK = "security_risk"

@dataclass
class SQLValidationReport:
    """Comprehensive SQL validation report."""
    is_valid: bool
    result: ValidationResult
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    performance_score: float
    estimated_execution_time: Optional[float] = None
    query_complexity: str = "unknown"
    tables_accessed: Set[str] = None
    columns_accessed: Set[str] = None
    
    def __post_init__(self):
        if self.tables_accessed is None:
            self.tables_accessed = set()
        if self.columns_accessed is None:
            self.columns_accessed = set()

@dataclass
class OptimizationSuggestion:
    """SQL optimization suggestion."""
    type: str
    priority: str  # high, medium, low
    description: str
    before: str
    after: str
    estimated_improvement: float

class SQLValidator:
    """Advanced SQL validation and optimization engine."""
    
    def __init__(self, engine: Engine, config: Dict[str, Any] = None):
        self.engine = engine
        self.config = config or {}
        self.inspector = inspect(engine)
        
        # Cache database schema information
        self.schema_cache = self._build_schema_cache()
        
        # SQL patterns for validation
        self.forbidden_patterns = [
            r'\bDROP\b',
            r'\bDELETE\b',
            r'\bUPDATE\b',
            r'\bINSERT\b',
            r'\bTRUNCATE\b',
            r'\bALTER\b',
            r'\bCREATE\b',
            r'\bREPLACE\b',
            r'\bGRANT\b',
            r'\bREVOKE\b'
        ]
        
        # Performance warning patterns
        self.performance_patterns = [
            r'SELECT\s+\*\s+FROM',
            r'WHERE.*LIKE\s+[\'"]%.*%[\'"]',
            r'WHERE.*NOT\s+IN\s*\(',
            r'WHERE.*OR.*OR.*OR',
            r'ORDER\s+BY.*RAND\(\)',
            r'GROUP\s+BY.*\d+'
        ]
        
        # Initialize statistics if available
        self.statistics = None
        if self.config.get('database', {}).get('enable_statistics_collection', False):
            self._load_database_statistics()
    
    def _build_schema_cache(self) -> Dict[str, Any]:
        """Build comprehensive schema cache."""
        schema_cache = {
            'tables': {},
            'columns': {},
            'indexes': {},
            'constraints': {}
        }
        
        try:
            # Get all tables
            for table_name in self.inspector.get_table_names():
                schema_cache['tables'][table_name.lower()] = table_name
                
                # Get columns for each table
                columns = self.inspector.get_columns(table_name)
                schema_cache['columns'][table_name.lower()] = {}
                
                for column in columns:
                    col_name = column['name'].lower()
                    schema_cache['columns'][table_name.lower()][col_name] = {
                        'name': column['name'],
                        'type': str(column['type']),
                        'nullable': column.get('nullable', True),
                        'default': column.get('default'),
                        'primary_key': column.get('primary_key', False)
                    }
                
                # Get indexes
                try:
                    indexes = self.inspector.get_indexes(table_name)
                    schema_cache['indexes'][table_name.lower()] = [
                        {
                            'name': idx['name'],
                            'columns': idx['column_names'],
                            'unique': idx.get('unique', False)
                        }
                        for idx in indexes
                    ]
                except Exception as e:
                    logger.debug(f"Could not get indexes for {table_name}: {e}")
                
                # Get foreign keys
                try:
                    fks = self.inspector.get_foreign_keys(table_name)
                    schema_cache['constraints'][table_name.lower()] = fks
                except Exception as e:
                    logger.debug(f"Could not get foreign keys for {table_name}: {e}")
            
            logger.info(f"Schema cache built for {len(schema_cache['tables'])} tables")
            return schema_cache
            
        except Exception as e:
            logger.error(f"Failed to build schema cache: {e}")
            return schema_cache
    
    def _load_database_statistics(self):
        """Load database statistics for optimization."""
        try:
            from .database_statistics import DatabaseStatisticsCollector
            self.statistics = DatabaseStatisticsCollector(self.engine)
            logger.info("Database statistics collector initialized")
        except ImportError:
            logger.warning("Database statistics module not available")
    
    def validate(self, sql_query: str) -> SQLValidationReport:
        """Comprehensive SQL validation."""
        report = SQLValidationReport(
            is_valid=True,
            result=ValidationResult.VALID,
            errors=[],
            warnings=[],
            suggestions=[],
            performance_score=100.0
        )
        
        try:
            # Basic syntax validation
            if not self._validate_syntax(sql_query, report):
                return report
            
            # Security validation
            if not self._validate_security(sql_query, report):
                return report
            
            # Schema validation
            if not self._validate_schema(sql_query, report):
                return report
            
            # Performance analysis
            self._analyze_performance(sql_query, report)
            
            # Generate optimization suggestions
            self._generate_optimization_suggestions(sql_query, report)
            
            # Calculate final validation result
            if report.errors:
                report.is_valid = False
                report.result = ValidationResult.INVALID_SCHEMA
            elif report.performance_score < 50:
                report.result = ValidationResult.PERFORMANCE_WARNING
            
            return report
            
        except Exception as e:
            report.is_valid = False
            report.result = ValidationResult.INVALID_SYNTAX
            report.errors.append(f"Validation failed: {str(e)}")
            return report
    
    def _validate_syntax(self, sql_query: str, report: SQLValidationReport) -> bool:
        """Validate SQL syntax."""
        if not SQLPARSE_AVAILABLE:
            logger.warning("sqlparse not available, skipping syntax validation")
            return True
        
        try:
            # Parse the SQL
            parsed = sqlparse.parse(sql_query)
            
            if not parsed:
                report.errors.append("Empty or invalid SQL query")
                return False
            
            # Check if it's a single statement
            statements = [stmt for stmt in parsed if stmt.ttype is None]
            if len(statements) > 1:
                report.warnings.append("Multiple statements detected, only the first will be executed")
            
            # Basic structural validation
            statement = statements[0]
            if not self._is_select_statement(statement):
                report.errors.append("Only SELECT statements are allowed")
                return False
            
            return True
            
        except Exception as e:
            report.errors.append(f"Syntax validation failed: {str(e)}")
            return False
    
    def _validate_security(self, sql_query: str, report: SQLValidationReport) -> bool:
        """Validate query for security risks."""
        query_upper = sql_query.upper()
        
        # Check for forbidden operations
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                report.errors.append(f"Forbidden operation detected: {pattern}")
                return False
        
        # Check for potential injection patterns
        injection_patterns = [
            r"'.*;\s*(DROP|DELETE|UPDATE|INSERT)",
            r"UNION\s+SELECT.*--",
            r"1\s*=\s*1",
            r"OR\s+1\s*=\s*1"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                report.warnings.append(f"Potential SQL injection pattern: {pattern}")
                report.performance_score -= 10
        
        return True
    
    def _validate_schema(self, sql_query: str, report: SQLValidationReport) -> bool:
        """Validate query against database schema."""
        if not SQLPARSE_AVAILABLE:
            return True
        
        try:
            parsed = sqlparse.parse(sql_query)[0]
            
            # Extract tables and columns
            tables, columns = self._extract_schema_elements(parsed)
            
            # Validate tables exist
            for table in tables:
                table_lower = table.lower()
                if table_lower not in self.schema_cache['tables']:
                    report.errors.append(f"Table '{table}' does not exist")
                    return False
                
                report.tables_accessed.add(table)
            
            # Validate columns exist
            for table, cols in columns.items():
                table_lower = table.lower()
                if table_lower in self.schema_cache['columns']:
                    table_columns = self.schema_cache['columns'][table_lower]
                    
                    for col in cols:
                        col_lower = col.lower()
                        if col != '*' and col_lower not in table_columns:
                            report.errors.append(f"Column '{col}' does not exist in table '{table}'")
                            return False
                        
                        if col != '*':
                            report.columns_accessed.add(f"{table}.{col}")
            
            return True
            
        except Exception as e:
            report.warnings.append(f"Schema validation incomplete: {str(e)}")
            return True  # Don't fail validation, just warn
    
    def _analyze_performance(self, sql_query: str, report: SQLValidationReport):
        """Analyze query performance and provide scoring."""
        query_upper = sql_query.upper()
        
        # Performance deductions
        deductions = 0
        
        # Check for performance anti-patterns
        for pattern in self.performance_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                if 'SELECT *' in pattern:
                    report.warnings.append("SELECT * can be inefficient, specify columns explicitly")
                    deductions += 15
                elif 'LIKE' in pattern:
                    report.warnings.append("Leading wildcard LIKE queries are slow")
                    deductions += 20
                elif 'NOT IN' in pattern:
                    report.warnings.append("NOT IN can be inefficient, consider NOT EXISTS")
                    deductions += 10
                elif 'OR.*OR.*OR' in pattern:
                    report.warnings.append("Multiple OR conditions can be slow, consider UNION")
                    deductions += 15
                elif 'RAND()' in pattern:
                    report.warnings.append("ORDER BY RAND() is very inefficient")
                    deductions += 30
        
        # Check for missing indexes (if statistics available)
        if self.statistics:
            index_warnings = self._check_index_usage(sql_query, report)
            deductions += len(index_warnings) * 10
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(sql_query)
        if complexity_score > 50:
            report.query_complexity = "complex"
            deductions += 10
        elif complexity_score > 25:
            report.query_complexity = "moderate"
            deductions += 5
        else:
            report.query_complexity = "simple"
        
        # Apply deductions
        report.performance_score = max(0, 100 - deductions)
        
        # Estimate execution time based on complexity and table sizes
        if self.statistics:
            report.estimated_execution_time = self._estimate_execution_time(sql_query, complexity_score)
    
    def _generate_optimization_suggestions(self, sql_query: str, report: SQLValidationReport):
        """Generate specific optimization suggestions."""
        suggestions = []
        
        # SELECT * optimization
        if re.search(r'SELECT\s+\*', sql_query, re.IGNORECASE):
            suggestions.append(
                "Replace SELECT * with specific column names to reduce data transfer and improve performance"
            )
        
        # LIMIT suggestion for large result sets
        if not re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE):
            suggestions.append(
                "Consider adding LIMIT clause to prevent accidentally large result sets"
            )
        
        # Index suggestions based on WHERE clauses
        where_columns = self._extract_where_columns(sql_query)
        for table, columns in where_columns.items():
            table_lower = table.lower()
            if table_lower in self.schema_cache['indexes']:
                indexed_columns = set()
                for index in self.schema_cache['indexes'][table_lower]:
                    indexed_columns.update(index['columns'])
                
                for col in columns:
                    if col.lower() not in [ic.lower() for ic in indexed_columns]:
                        suggestions.append(
                            f"Consider creating an index on {table}.{col} to improve WHERE clause performance"
                        )
        
        report.suggestions.extend(suggestions)
    
    def _extract_schema_elements(self, parsed_query) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """Extract tables and columns from parsed SQL."""
        tables = set()
        columns = {}
        
        def extract_from_token(token, current_table=None):
            if hasattr(token, 'tokens'):
                for subtoken in token.tokens:
                    extract_from_token(subtoken, current_table)
            else:
                if token.ttype in (tokens.Name, None) and isinstance(token.value, str):
                    value = token.value.strip()
                    
                    # Simple heuristic: if it follows FROM or JOIN, it's likely a table
                    # This is a simplified approach - real SQL parsing is much more complex
                    if value and not value.lower() in ('select', 'from', 'where', 'and', 'or', 'as'):
                        # Add to tables set (simplified detection)
                        if len(value.split('.')) == 1 and not value.startswith("'"):
                            tables.add(value)
        
        extract_from_token(parsed_query)
        
        # Simplified column extraction - in practice, this would need more sophisticated parsing
        for table in tables:
            columns[table] = {'*'}  # Simplified - assume all queries use *
        
        return tables, columns
    
    def _extract_where_columns(self, sql_query: str) -> Dict[str, Set[str]]:
        """Extract columns used in WHERE clauses."""
        where_columns = {}
        
        # Simplified WHERE clause parsing
        where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', 
                               sql_query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column references (table.column or just column)
            column_pattern = r'\b(\w+)\.(\w+)\b|\b(\w+)\s*(?:=|<|>|LIKE|IN)'
            matches = re.findall(column_pattern, where_clause, re.IGNORECASE)
            
            for match in matches:
                if match[0] and match[1]:  # table.column format
                    table, column = match[0], match[1]
                    if table not in where_columns:
                        where_columns[table] = set()
                    where_columns[table].add(column)
                elif match[2]:  # just column format
                    # Would need context to determine which table
                    pass
        
        return where_columns
    
    def _calculate_complexity(self, sql_query: str) -> float:
        """Calculate query complexity score."""
        complexity = 0
        query_upper = sql_query.upper()
        
        # Count various complexity indicators
        complexity += len(re.findall(r'\bJOIN\b', query_upper)) * 10
        complexity += len(re.findall(r'\bSUBQUERY\b|\bEXISTS\b', query_upper)) * 15
        complexity += len(re.findall(r'\bUNION\b', query_upper)) * 12
        complexity += len(re.findall(r'\bGROUP BY\b', query_upper)) * 8
        complexity += len(re.findall(r'\bHAVING\b', query_upper)) * 10
        complexity += len(re.findall(r'\bORDER BY\b', query_upper)) * 5
        complexity += len(re.findall(r'\bWHERE\b', query_upper)) * 3
        
        # Function complexity
        complexity += len(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\b', query_upper)) * 5
        
        return complexity
    
    def _estimate_execution_time(self, sql_query: str, complexity_score: float) -> float:
        """Estimate query execution time."""
        if not self.statistics:
            return None
        
        # Base time (milliseconds)
        base_time = 10.0
        
        # Complexity multiplier
        complexity_multiplier = 1 + (complexity_score / 100)
        
        # Table size multiplier (simplified)
        table_multiplier = 1.0
        for table in self.schema_cache['tables']:
            if table.upper() in sql_query.upper():
                # Would use actual row counts from statistics
                table_multiplier *= 1.5  # Simplified multiplier
        
        estimated_time = base_time * complexity_multiplier * table_multiplier
        return round(estimated_time, 2)
    
    def _check_index_usage(self, sql_query: str, report: SQLValidationReport) -> List[str]:
        """Check for potential missing indexes."""
        warnings = []
        
        # This would integrate with database statistics to check actual index usage
        # For now, provide basic suggestions
        
        if re.search(r'\bWHERE\b.*\bLIKE\b', sql_query, re.IGNORECASE):
            warnings.append("LIKE operations may benefit from full-text indexes")
        
        return warnings
    
    def _is_select_statement(self, statement) -> bool:
        """Check if the statement is a SELECT statement."""
        if not SQLPARSE_AVAILABLE:
            return 'SELECT' in statement.upper()
        
        # Find the first meaningful token
        for token in statement.flatten():
            if token.ttype in (tokens.Keyword, tokens.Keyword.DML):
                return token.value.upper() == 'SELECT'
        
        return False
    
    def optimize_query(self, sql_query: str) -> Tuple[str, List[OptimizationSuggestion]]:
        """Automatically optimize SQL query where possible."""
        optimized_query = sql_query
        suggestions = []
        
        # Automatic optimizations that can be safely applied
        
        # 1. Add explicit column limits if using SELECT *
        if re.search(r'SELECT\s+\*\s+FROM\s+(\w+)', optimized_query, re.IGNORECASE):
            # This would need schema information to replace * with actual columns
            pass  # Skip for now as it requires careful implementation
        
        # 2. Add LIMIT if missing and no aggregation
        if (not re.search(r'\bLIMIT\b', optimized_query, re.IGNORECASE) and 
            not re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP)\b', optimized_query, re.IGNORECASE)):
            
            limit_clause = " LIMIT 1000"  # Default reasonable limit
            optimized_query += limit_clause
            
            suggestions.append(OptimizationSuggestion(
                type="limit_addition",
                priority="medium",
                description="Added LIMIT clause to prevent large result sets",
                before=sql_query,
                after=optimized_query,
                estimated_improvement=20.0
            ))
        
        # 3. Rewrite NOT IN as NOT EXISTS where beneficial
        not_in_pattern = r'(\w+)\s+NOT\s+IN\s*\([^)]+\)'
        if re.search(not_in_pattern, optimized_query, re.IGNORECASE):
            # This would need careful rewriting - skip for now
            pass
        
        return optimized_query, suggestions
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation capabilities and statistics."""
        return {
            'validator_version': '1.0.0',
            'capabilities': {
                'syntax_validation': SQLPARSE_AVAILABLE,
                'schema_validation': True,
                'security_validation': True,
                'performance_analysis': True,
                'optimization_suggestions': True,
                'statistics_integration': self.statistics is not None
            },
            'schema_cache': {
                'tables_count': len(self.schema_cache['tables']),
                'total_columns': sum(len(cols) for cols in self.schema_cache['columns'].values()),
                'indexes_tracked': sum(len(indexes) for indexes in self.schema_cache['indexes'].values())
            },
            'validation_patterns': {
                'forbidden_operations': len(self.forbidden_patterns),
                'performance_checks': len(self.performance_patterns)
            }
        }

# Global validator instance
_global_validator = None

def get_sql_validator(engine: Engine, config: Dict[str, Any] = None) -> SQLValidator:
    """Get global SQL validator instance."""
    global _global_validator
    
    if _global_validator is None or _global_validator.engine != engine:
        _global_validator = SQLValidator(engine, config)
    
    return _global_validator

def reset_sql_validator():
    """Reset global SQL validator."""
    global _global_validator
    _global_validator = None
