# src/core/sql/sql_optimizer.py

import json
import logging
import re
import sqlite3
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
import pandas as pd

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization levels for query processing."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    AGGRESSIVE = "aggressive"

class QueryComplexity(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

@dataclass
class ExecutionPlan:
    """Represents a query execution plan."""
    raw_plan: Dict[str, Any]
    estimated_cost: float
    estimated_rows: int
    execution_time: Optional[float] = None
    scan_operations: List[str] = field(default_factory=list)
    join_operations: List[str] = field(default_factory=list)
    index_usage: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)

@dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion with before/after comparison."""
    type: str
    priority: str  # high, medium, low
    description: str
    original_query: str
    optimized_query: str
    estimated_improvement_percent: float
    reasoning: str
    applies_to: List[str] = field(default_factory=list)  # table names affected

@dataclass
class QueryOptimizationResult:
    """Complete optimization analysis result."""
    original_query: str
    optimized_query: str
    original_plan: ExecutionPlan
    optimized_plan: ExecutionPlan
    suggestions: List[OptimizationSuggestion]
    improvement_summary: Dict[str, Any]
    confidence: float
    safe_to_auto_apply: bool

class SQLQueryOptimizer:
    """
    Advanced SQL Query Optimizer with EXPLAIN plan analysis, 
    query rewriting, and cost-based optimization.
    """
    
    def __init__(self, engine: Engine, config: Dict[str, Any] = None):
        self.engine = engine
        self.config = config or {}
        self.inspector = inspect(engine)
        
        # Initialize database-specific settings
        self.db_type = self._detect_database_type()
        self.schema_cache = self._build_schema_cache()
        
        # Optimization patterns and rules
        self.optimization_rules = self._load_optimization_rules()
        
        # Performance tracking
        self.optimization_history: List[QueryOptimizationResult] = []
        
        logger.info(f"SQL Optimizer initialized for {self.db_type} database")

    def _detect_database_type(self) -> str:
        """Detect the type of database we're working with."""
        dialect = self.engine.dialect.name.lower()
        if 'sqlite' in dialect:
            return 'sqlite'
        elif 'postgresql' in dialect:
            return 'postgresql'
        elif 'mysql' in dialect:
            return 'mysql'
        else:
            return 'unknown'

    def _build_schema_cache(self) -> Dict[str, Any]:
        """Build comprehensive schema information cache."""
        schema_cache = {
            'tables': {},
            'columns': {},
            'indexes': {},
            'statistics': {},
            'foreign_keys': {}
        }
        
        try:
            for table_name in self.inspector.get_table_names():
                # Table information
                schema_cache['tables'][table_name.lower()] = {
                    'name': table_name,
                    'row_count': self._estimate_table_rows(table_name)
                }
                
                # Column information
                columns = self.inspector.get_columns(table_name)
                schema_cache['columns'][table_name.lower()] = {}
                
                for col in columns:
                    schema_cache['columns'][table_name.lower()][col['name'].lower()] = {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True),
                        'primary_key': col.get('primary_key', False),
                        'selectivity': self._estimate_column_selectivity(table_name, col['name'])
                    }
                
                # Index information
                try:
                    indexes = self.inspector.get_indexes(table_name)
                    schema_cache['indexes'][table_name.lower()] = []
                    
                    for idx in indexes:
                        schema_cache['indexes'][table_name.lower()].append({
                            'name': idx['name'],
                            'columns': idx['column_names'],
                            'unique': idx.get('unique', False),
                            'selectivity': self._estimate_index_selectivity(table_name, idx['column_names'])
                        })
                        
                except Exception as e:
                    logger.debug(f"Could not get indexes for {table_name}: {e}")
                
                # Foreign key relationships
                try:
                    fks = self.inspector.get_foreign_keys(table_name)
                    schema_cache['foreign_keys'][table_name.lower()] = fks
                except Exception as e:
                    logger.debug(f"Could not get foreign keys for {table_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Schema cache building failed: {e}")
            
        return schema_cache

    def _estimate_table_rows(self, table_name: str) -> int:
        """Estimate number of rows in a table."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar() or 0
        except Exception:
            return 1000  # Default estimate

    def _estimate_column_selectivity(self, table_name: str, column_name: str) -> float:
        """Estimate column selectivity (uniqueness ratio)."""
        try:
            with self.engine.connect() as conn:
                # Get total rows and unique values
                total_query = text(f"SELECT COUNT(*) FROM {table_name}")
                unique_query = text(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}")
                
                total_rows = conn.execute(total_query).scalar() or 1
                unique_values = conn.execute(unique_query).scalar() or 1
                
                return unique_values / total_rows if total_rows > 0 else 0.0
                
        except Exception:
            return 0.1  # Default moderate selectivity

    def _estimate_index_selectivity(self, table_name: str, columns: List[str]) -> float:
        """Estimate selectivity of an index on given columns."""
        try:
            if not columns:
                return 0.0
                
            with self.engine.connect() as conn:
                # Estimate selectivity as unique combinations / total rows
                total_query = text(f"SELECT COUNT(*) FROM {table_name}")
                
                # Build unique combination query
                column_list = ', '.join(columns)
                unique_query = text(f"SELECT COUNT(DISTINCT {column_list}) FROM {table_name}")
                
                total_rows = conn.execute(total_query).scalar() or 1
                unique_combinations = conn.execute(unique_query).scalar() or 1
                
                return unique_combinations / total_rows if total_rows > 0 else 0.0
                
        except Exception:
            return 0.1  # Default selectivity

    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules and patterns."""
        return {
            'index_recommendations': [
                {
                    'pattern': r'WHERE\s+(\w+\.\w+|\w+)\s*=',
                    'type': 'equality_filter',
                    'suggestion': 'Consider adding index on equality filter column'
                },
                {
                    'pattern': r'WHERE\s+(\w+\.\w+|\w+)\s+LIKE\s+\'([^%].*?)%\'',
                    'type': 'prefix_search',
                    'suggestion': 'Consider adding index for prefix search optimization'
                },
                {
                    'pattern': r'ORDER\s+BY\s+(\w+(?:\.\w+)?(?:\s*,\s*\w+(?:\.\w+)?)*)',
                    'type': 'order_by',
                    'suggestion': 'Consider adding index on ORDER BY columns'
                }
            ],
            'query_rewrites': [
                {
                    'pattern': r'SELECT\s+\*\s+FROM',
                    'replacement': 'SELECT {specific_columns} FROM',
                    'type': 'select_star_elimination',
                    'benefit': 'Reduces I/O by selecting only needed columns'
                },
                {
                    'pattern': r'WHERE\s+(\w+)\s+IN\s*\(\s*SELECT',
                    'replacement': 'WHERE EXISTS (SELECT 1 FROM',
                    'type': 'in_to_exists',
                    'benefit': 'EXISTS can be more efficient than IN subqueries'
                },
                {
                    'pattern': r'WHERE\s+(\w+)\s+NOT\s+IN\s*\(',
                    'replacement': 'WHERE NOT EXISTS (',
                    'type': 'not_in_to_not_exists', 
                    'benefit': 'NOT EXISTS handles NULL values better and can be faster'
                }
            ],
            'join_optimizations': [
                {
                    'type': 'join_order',
                    'rule': 'Place most selective tables first in join order',
                    'implementation': 'analyze_join_selectivity'
                },
                {
                    'type': 'join_type_optimization',
                    'rule': 'Use appropriate join types (INNER vs LEFT)',
                    'implementation': 'optimize_join_types'
                }
            ]
        }

    def analyze_execution_plan(self, sql_query: str) -> ExecutionPlan:
        """Analyze query execution plan using database-specific EXPLAIN."""
        try:
            with self.engine.connect() as conn:
                # Get execution plan
                if self.db_type == 'sqlite':
                    plan_result = conn.execute(text(f"EXPLAIN QUERY PLAN {sql_query}"))
                    raw_plan = [dict(row._mapping) for row in plan_result]
                    
                    return self._parse_sqlite_plan(raw_plan, sql_query)
                    
                elif self.db_type == 'postgresql':
                    plan_result = conn.execute(text(f"EXPLAIN (FORMAT JSON, ANALYZE TRUE) {sql_query}"))
                    raw_plan = plan_result.scalar()
                    
                    return self._parse_postgresql_plan(raw_plan, sql_query)
                    
                elif self.db_type == 'mysql':
                    plan_result = conn.execute(text(f"EXPLAIN FORMAT=JSON {sql_query}"))
                    raw_plan = json.loads(plan_result.scalar())
                    
                    return self._parse_mysql_plan(raw_plan, sql_query)
                    
                else:
                    # Fallback: basic analysis without actual EXPLAIN
                    return self._analyze_query_without_explain(sql_query)
                    
        except Exception as e:
            logger.warning(f"Execution plan analysis failed: {e}")
            return self._analyze_query_without_explain(sql_query)

    def _parse_sqlite_plan(self, raw_plan: List[Dict], sql_query: str) -> ExecutionPlan:
        """Parse SQLite EXPLAIN QUERY PLAN output."""
        scan_operations = []
        join_operations = []
        index_usage = []
        bottlenecks = []
        optimization_opportunities = []
        
        estimated_cost = 0.0
        estimated_rows = 0
        
        for step in raw_plan:
            detail = step.get('detail', '').lower()
            
            # Analyze scan types
            if 'scan table' in detail:
                if 'using index' in detail:
                    index_name = re.search(r'using index (\w+)', detail)
                    if index_name:
                        index_usage.append(index_name.group(1))
                        scan_operations.append(f"Index scan on {index_name.group(1)}")
                        estimated_cost += 1.0  # Index scan cost
                    else:
                        scan_operations.append("Index scan")
                        estimated_cost += 1.0
                else:
                    table_match = re.search(r'scan table (\w+)', detail)
                    if table_match:
                        table_name = table_match.group(1)
                        scan_operations.append(f"Full table scan on {table_name}")
                        bottlenecks.append(f"Full table scan on {table_name}")
                        
                        # Estimate cost based on table size
                        table_rows = self.schema_cache.get('tables', {}).get(table_name.lower(), {}).get('row_count', 1000)
                        estimated_cost += table_rows * 0.001  # Cost per row
                        estimated_rows += table_rows
                        
                        # Suggest optimization
                        optimization_opportunities.append(f"Consider adding index to avoid full table scan on {table_name}")
            
            # Analyze joins
            elif 'join' in detail:
                join_operations.append(detail)
                estimated_cost += 10.0  # Basic join cost
                
                if 'nested loop' in detail:
                    bottlenecks.append("Nested loop join detected - may be inefficient for large tables")
        
        return ExecutionPlan(
            raw_plan=raw_plan,
            estimated_cost=estimated_cost,
            estimated_rows=estimated_rows,
            scan_operations=scan_operations,
            join_operations=join_operations,
            index_usage=index_usage,
            bottlenecks=bottlenecks,
            optimization_opportunities=optimization_opportunities
        )

    def _parse_postgresql_plan(self, raw_plan: str, sql_query: str) -> ExecutionPlan:
        """Parse PostgreSQL EXPLAIN JSON output."""
        try:
            plan_data = json.loads(raw_plan)[0]['Plan']
            
            return ExecutionPlan(
                raw_plan=plan_data,
                estimated_cost=plan_data.get('Total Cost', 0.0),
                estimated_rows=plan_data.get('Plan Rows', 0),
                execution_time=plan_data.get('Actual Total Time'),
                scan_operations=self._extract_postgres_scans(plan_data),
                join_operations=self._extract_postgres_joins(plan_data),
                index_usage=self._extract_postgres_indexes(plan_data),
                bottlenecks=self._identify_postgres_bottlenecks(plan_data),
                optimization_opportunities=self._suggest_postgres_optimizations(plan_data)
            )
            
        except Exception as e:
            logger.warning(f"PostgreSQL plan parsing failed: {e}")
            return self._analyze_query_without_explain(sql_query)

    def _parse_mysql_plan(self, raw_plan: Dict, sql_query: str) -> ExecutionPlan:
        """Parse MySQL EXPLAIN JSON output."""
        try:
            query_block = raw_plan['query_block']
            
            estimated_cost = query_block.get('cost_info', {}).get('query_cost', 0.0)
            estimated_rows = query_block.get('cost_info', {}).get('read_cost', 0)
            
            return ExecutionPlan(
                raw_plan=raw_plan,
                estimated_cost=float(estimated_cost),
                estimated_rows=int(estimated_rows),
                scan_operations=self._extract_mysql_scans(query_block),
                join_operations=self._extract_mysql_joins(query_block),
                index_usage=self._extract_mysql_indexes(query_block),
                bottlenecks=self._identify_mysql_bottlenecks(query_block),
                optimization_opportunities=self._suggest_mysql_optimizations(query_block)
            )
            
        except Exception as e:
            logger.warning(f"MySQL plan parsing failed: {e}")
            return self._analyze_query_without_explain(sql_query)

    def _analyze_query_without_explain(self, sql_query: str) -> ExecutionPlan:
        """Fallback analysis when EXPLAIN is not available."""
        # Basic static analysis
        query_upper = sql_query.upper()
        
        scan_operations = []
        join_operations = []
        bottlenecks = []
        optimization_opportunities = []
        
        # Look for table scans
        table_matches = re.findall(r'FROM\s+(\w+)', query_upper)
        for table in table_matches:
            table_rows = self.schema_cache.get('tables', {}).get(table.lower(), {}).get('row_count', 1000)
            
            if table_rows > 10000:
                scan_operations.append(f"Estimated large table scan on {table}")
                bottlenecks.append(f"Large table {table} may cause performance issues")
        
        # Look for joins
        if 'JOIN' in query_upper:
            join_count = len(re.findall(r'JOIN', query_upper))
            join_operations.append(f"Estimated {join_count} join operations")
            
            if join_count > 3:
                bottlenecks.append(f"Complex query with {join_count} joins")
                optimization_opportunities.append("Consider breaking complex joins into simpler queries")
        
        # Basic cost estimation
        estimated_cost = len(table_matches) * 10.0 + len(re.findall(r'JOIN', query_upper)) * 5.0
        estimated_rows = sum([
            self.schema_cache.get('tables', {}).get(table.lower(), {}).get('row_count', 1000)
            for table in table_matches
        ])
        
        return ExecutionPlan(
            raw_plan={'analysis': 'static_analysis', 'query': sql_query},
            estimated_cost=estimated_cost,
            estimated_rows=estimated_rows,
            scan_operations=scan_operations,
            join_operations=join_operations,
            index_usage=[],
            bottlenecks=bottlenecks,
            optimization_opportunities=optimization_opportunities
        )

    def optimize_query(self, sql_query: str, optimization_level: OptimizationLevel = OptimizationLevel.INTERMEDIATE) -> QueryOptimizationResult:
        """
        Main optimization method that analyzes and optimizes a SQL query.
        
        Args:
            sql_query: The SQL query to optimize
            optimization_level: Level of optimization to apply
            
        Returns:
            Complete optimization result with suggestions and analysis
        """
        logger.info(f"Starting query optimization with level: {optimization_level.value}")
        
        # Analyze original query
        original_plan = self.analyze_execution_plan(sql_query)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(sql_query, original_plan)
        
        # Apply optimizations based on level
        optimized_query = self._apply_optimizations(sql_query, suggestions, optimization_level)
        
        # Analyze optimized query
        optimized_plan = self.analyze_execution_plan(optimized_query)
        
        # Calculate improvement metrics
        improvement_summary = self._calculate_improvement_metrics(original_plan, optimized_plan)
        
        # Determine confidence and safety
        confidence = self._calculate_optimization_confidence(suggestions, improvement_summary)
        safe_to_auto_apply = self._is_safe_to_auto_apply(suggestions, confidence)
        
        result = QueryOptimizationResult(
            original_query=sql_query,
            optimized_query=optimized_query,
            original_plan=original_plan,
            optimized_plan=optimized_plan,
            suggestions=suggestions,
            improvement_summary=improvement_summary,
            confidence=confidence,
            safe_to_auto_apply=safe_to_auto_apply
        )
        
        # Store in optimization history
        self.optimization_history.append(result)
        
        logger.info(f"Query optimization completed with {len(suggestions)} suggestions")
        return result

    def _generate_optimization_suggestions(self, sql_query: str, execution_plan: ExecutionPlan) -> List[OptimizationSuggestion]:
        """Generate comprehensive optimization suggestions."""
        suggestions = []
        
        # 1. Index-based optimizations
        suggestions.extend(self._suggest_index_optimizations(sql_query, execution_plan))
        
        # 2. Query rewrite optimizations
        suggestions.extend(self._suggest_query_rewrites(sql_query))
        
        # 3. Join optimizations
        suggestions.extend(self._suggest_join_optimizations(sql_query, execution_plan))
        
        # 4. Predicate optimizations
        suggestions.extend(self._suggest_predicate_optimizations(sql_query))
        
        # 5. Structure optimizations
        suggestions.extend(self._suggest_structure_optimizations(sql_query, execution_plan))
        
        return suggestions

    def _suggest_index_optimizations(self, sql_query: str, execution_plan: ExecutionPlan) -> List[OptimizationSuggestion]:
        """Suggest index-based optimizations."""
        suggestions = []
        
        # Analyze WHERE clause columns
        where_columns = self._extract_where_columns(sql_query)
        
        for table, columns in where_columns.items():
            table_lower = table.lower()
            
            if table_lower in self.schema_cache['indexes']:
                existing_indexes = self.schema_cache['indexes'][table_lower]
                existing_index_columns = set()
                
                for idx in existing_indexes:
                    existing_index_columns.update([col.lower() for col in idx['columns']])
                
                # Check for missing indexes on WHERE columns
                for col in columns:
                    if col.lower() not in existing_index_columns:
                        suggestions.append(OptimizationSuggestion(
                            type='missing_index',
                            priority='high',
                            description=f'Add index on {table}.{col} to improve WHERE clause performance',
                            original_query=sql_query,
                            optimized_query=sql_query,  # Index creation doesn't change query
                            estimated_improvement_percent=30.0,
                            reasoning=f'Column {col} used in WHERE clause but not indexed',
                            applies_to=[table]
                        ))
        
        # Analyze ORDER BY columns
        order_by_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s*$)', sql_query, re.IGNORECASE)
        if order_by_match:
            order_columns = [col.strip() for col in order_by_match.group(1).split(',')]
            
            for col_expr in order_columns:
                # Extract table.column or just column
                col_match = re.match(r'(?:(\w+)\.)?(\w+)', col_expr.strip())
                if col_match:
                    table_name = col_match.group(1) or 'default_table'
                    col_name = col_match.group(2)
                    
                    suggestions.append(OptimizationSuggestion(
                        type='order_by_index',
                        priority='medium',
                        description=f'Consider index on ORDER BY column {col_name}',
                        original_query=sql_query,
                        optimized_query=sql_query,
                        estimated_improvement_percent=20.0,
                        reasoning='ORDER BY operations benefit from indexes on sorted columns',
                        applies_to=[table_name] if table_name != 'default_table' else []
                    ))
        
        return suggestions

    def _suggest_query_rewrites(self, sql_query: str) -> List[OptimizationSuggestion]:
        """Suggest query rewrite optimizations."""
        suggestions = []
        
        # 1. SELECT * optimization
        if re.search(r'SELECT\s+\*', sql_query, re.IGNORECASE):
            # Try to determine which columns are actually needed
            optimized = re.sub(r'SELECT\s+\*', 'SELECT column1, column2, column3', sql_query, flags=re.IGNORECASE)
            
            suggestions.append(OptimizationSuggestion(
                type='select_star_elimination',
                priority='high',
                description='Replace SELECT * with specific columns to reduce I/O',
                original_query=sql_query,
                optimized_query=optimized,
                estimated_improvement_percent=25.0,
                reasoning='SELECT * transfers unnecessary data and prevents some optimizations'
            ))
        
        # 2. NOT IN to NOT EXISTS
        not_in_pattern = r'WHERE\s+(\w+(?:\.\w+)?)\s+NOT\s+IN\s*\('
        if re.search(not_in_pattern, sql_query, re.IGNORECASE):
            optimized = re.sub(
                not_in_pattern,
                r'WHERE NOT EXISTS (SELECT 1 FROM subquery_table WHERE subquery_table.column = \1 AND',
                sql_query,
                flags=re.IGNORECASE
            )
            
            suggestions.append(OptimizationSuggestion(
                type='not_in_to_not_exists',
                priority='medium',
                description='Convert NOT IN to NOT EXISTS for better NULL handling and performance',
                original_query=sql_query,
                optimized_query=optimized,
                estimated_improvement_percent=15.0,
                reasoning='NOT EXISTS is more efficient and handles NULLs correctly'
            ))
        
        # 3. Subquery to JOIN optimization
        subquery_pattern = r'WHERE\s+(\w+(?:\.\w+)?)\s+IN\s*\(\s*SELECT\s+(\w+)'
        if re.search(subquery_pattern, sql_query, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                type='subquery_to_join',
                priority='medium',
                description='Consider converting IN subquery to JOIN for better performance',
                original_query=sql_query,
                optimized_query=sql_query,  # Would need more complex rewriting
                estimated_improvement_percent=20.0,
                reasoning='JOINs are often more efficient than correlated subqueries'
            ))
        
        return suggestions

    def _suggest_join_optimizations(self, sql_query: str, execution_plan: ExecutionPlan) -> List[OptimizationSuggestion]:
        """Suggest join-related optimizations."""
        suggestions = []
        
        # Analyze join order based on table sizes
        join_tables = re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE)
        
        if len(join_tables) > 1:
            # Get table sizes
            table_sizes = {}
            for table in join_tables:
                table_info = self.schema_cache.get('tables', {}).get(table.lower(), {})
                table_sizes[table] = table_info.get('row_count', 1000)
            
            # Sort by size (smallest first is often optimal)
            sorted_tables = sorted(table_sizes.items(), key=lambda x: x[1])
            
            if [t[0] for t in sorted_tables] != join_tables:
                suggestions.append(OptimizationSuggestion(
                    type='join_order_optimization',
                    priority='high',
                    description='Optimize join order by placing smaller tables first',
                    original_query=sql_query,
                    optimized_query=sql_query,  # Would need complex reordering
                    estimated_improvement_percent=35.0,
                    reasoning='Joining smaller tables first reduces intermediate result sizes',
                    applies_to=[t[0] for t in sorted_tables]
                ))
        
        return suggestions

    def _suggest_predicate_optimizations(self, sql_query: str) -> List[OptimizationSuggestion]:
        """Suggest predicate and WHERE clause optimizations."""
        suggestions = []
        
        # 1. Leading wildcard LIKE optimization
        if re.search(r'LIKE\s+\'%.*?\'', sql_query, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                type='leading_wildcard_optimization',
                priority='high',
                description='Leading wildcards in LIKE prevent index usage',
                original_query=sql_query,
                optimized_query=sql_query,  # Would need full-text search rewrite
                estimated_improvement_percent=40.0,
                reasoning='Leading wildcards force full table scans'
            ))
        
        # 2. Function in WHERE clause
        function_pattern = r'WHERE\s+\w+\([^)]+\)\s*[=<>]'
        if re.search(function_pattern, sql_query, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                type='function_in_where',
                priority='medium',
                description='Functions in WHERE clause prevent index usage',
                original_query=sql_query,
                optimized_query=sql_query,
                estimated_improvement_percent=25.0,
                reasoning='Functions on columns in WHERE clauses make indexes unusable'
            ))
        
        return suggestions

    def _suggest_structure_optimizations(self, sql_query: str, execution_plan: ExecutionPlan) -> List[OptimizationSuggestion]:
        """Suggest structural optimizations."""
        suggestions = []
        
        # 1. LIMIT optimization
        if not re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE) and execution_plan.estimated_rows > 10000:
            optimized = sql_query.rstrip(';') + ' LIMIT 1000'
            
            suggestions.append(OptimizationSuggestion(
                type='add_limit',
                priority='low',
                description='Add LIMIT clause to prevent large result sets',
                original_query=sql_query,
                optimized_query=optimized,
                estimated_improvement_percent=10.0,
                reasoning='Large result sets can consume excessive memory and network bandwidth'
            ))
        
        # 2. DISTINCT optimization
        if re.search(r'SELECT\s+DISTINCT', sql_query, re.IGNORECASE):
            suggestions.append(OptimizationSuggestion(
                type='distinct_optimization',
                priority='low',
                description='DISTINCT can be expensive - ensure it\'s necessary',
                original_query=sql_query,
                optimized_query=sql_query,
                estimated_improvement_percent=5.0,
                reasoning='DISTINCT requires sorting/grouping - avoid if not needed'
            ))
        
        return suggestions

    def _apply_optimizations(self, sql_query: str, suggestions: List[OptimizationSuggestion], level: OptimizationLevel) -> str:
        """Apply optimizations based on the specified level."""
        optimized_query = sql_query
        
        # Determine which optimizations to apply based on level
        if level == OptimizationLevel.BASIC:
            # Apply only high-priority, safe optimizations
            applicable_suggestions = [s for s in suggestions if s.priority == 'high' and 'safe' in s.type]
        elif level == OptimizationLevel.INTERMEDIATE:
            # Apply high and medium priority optimizations
            applicable_suggestions = [s for s in suggestions if s.priority in ['high', 'medium']]
        else:  # AGGRESSIVE
            # Apply all optimizations
            applicable_suggestions = suggestions
        
        # Apply optimizations (simplified - real implementation would be more complex)
        for suggestion in applicable_suggestions:
            if suggestion.optimized_query != suggestion.original_query:
                if suggestion.type == 'select_star_elimination':
                    # This would need actual schema information to replace with real columns
                    pass
                elif suggestion.type == 'add_limit':
                    optimized_query = suggestion.optimized_query
                # Add more optimization applications as needed
        
        return optimized_query

    def _extract_where_columns(self, sql_query: str) -> Dict[str, List[str]]:
        """Extract columns used in WHERE clauses."""
        where_columns = {}
        
        # Find WHERE clause
        where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', 
                               sql_query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column references
            column_patterns = [
                r'\b(\w+)\.(\w+)\s*[=<>!]',  # table.column comparisons
                r'\b(\w+)\s*[=<>!]'          # column comparisons
            ]
            
            for pattern in column_patterns:
                matches = re.findall(pattern, where_clause, re.IGNORECASE)
                
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        table, column = match
                        if table not in where_columns:
                            where_columns[table] = []
                        where_columns[table].append(column)
                    elif isinstance(match, str):
                        # Column without table prefix - assign to default
                        if 'default_table' not in where_columns:
                            where_columns['default_table'] = []
                        where_columns['default_table'].append(match)
        
        return where_columns

    def _calculate_improvement_metrics(self, original_plan: ExecutionPlan, optimized_plan: ExecutionPlan) -> Dict[str, Any]:
        """Calculate improvement metrics between original and optimized plans."""
        cost_improvement = 0.0
        if original_plan.estimated_cost > 0:
            cost_improvement = ((original_plan.estimated_cost - optimized_plan.estimated_cost) / 
                              original_plan.estimated_cost) * 100
        
        return {
            'cost_improvement_percent': cost_improvement,
            'estimated_original_cost': original_plan.estimated_cost,
            'estimated_optimized_cost': optimized_plan.estimated_cost,
            'original_bottlenecks': len(original_plan.bottlenecks),
            'optimized_bottlenecks': len(optimized_plan.bottlenecks),
            'bottlenecks_resolved': len(original_plan.bottlenecks) - len(optimized_plan.bottlenecks)
        }

    def _calculate_optimization_confidence(self, suggestions: List[OptimizationSuggestion], improvement_summary: Dict[str, Any]) -> float:
        """Calculate confidence score for optimizations."""
        if not suggestions:
            return 0.0
        
        # Base confidence on number and quality of suggestions
        high_priority_count = len([s for s in suggestions if s.priority == 'high'])
        medium_priority_count = len([s for s in suggestions if s.priority == 'medium'])
        
        # Weight by priority
        confidence = (high_priority_count * 0.8 + medium_priority_count * 0.5) / len(suggestions)
        
        # Adjust based on estimated improvement
        if improvement_summary.get('cost_improvement_percent', 0) > 20:
            confidence += 0.2
        elif improvement_summary.get('cost_improvement_percent', 0) > 10:
            confidence += 0.1
        
        return min(1.0, confidence)

    def _is_safe_to_auto_apply(self, suggestions: List[OptimizationSuggestion], confidence: float) -> bool:
        """Determine if optimizations are safe to auto-apply."""
        # Only auto-apply if high confidence and no risky optimizations
        risky_types = ['subquery_to_join', 'join_order_optimization', 'not_in_to_not_exists']
        
        has_risky_optimizations = any(s.type in risky_types for s in suggestions)
        
        return confidence > 0.7 and not has_risky_optimizations

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization capabilities and history."""
        return {
            'database_type': self.db_type,
            'total_optimizations_performed': len(self.optimization_history),
            'average_improvement': sum([
                r.improvement_summary.get('cost_improvement_percent', 0) 
                for r in self.optimization_history
            ]) / len(self.optimization_history) if self.optimization_history else 0,
            'schema_tables_analyzed': len(self.schema_cache['tables']),
            'optimization_rules_loaded': len(self.optimization_rules),
            'capabilities': [
                'explain_plan_analysis',
                'cost_based_optimization', 
                'index_recommendations',
                'query_rewriting',
                'join_optimization',
                'predicate_optimization'
            ]
        }

    # Database-specific helper methods
    def _extract_postgres_scans(self, plan_data: Dict) -> List[str]:
        """Extract scan operations from PostgreSQL plan."""
        scans = []
        node_type = plan_data.get('Node Type', '')
        
        if 'Scan' in node_type:
            scans.append(node_type)
        
        # Recursively check child plans
        if 'Plans' in plan_data:
            for child_plan in plan_data['Plans']:
                scans.extend(self._extract_postgres_scans(child_plan))
        
        return scans

    def _extract_postgres_joins(self, plan_data: Dict) -> List[str]:
        """Extract join operations from PostgreSQL plan."""
        joins = []
        node_type = plan_data.get('Node Type', '')
        
        if 'Join' in node_type:
            joins.append(node_type)
        
        if 'Plans' in plan_data:
            for child_plan in plan_data['Plans']:
                joins.extend(self._extract_postgres_joins(child_plan))
        
        return joins

    def _extract_postgres_indexes(self, plan_data: Dict) -> List[str]:
        """Extract index usage from PostgreSQL plan."""
        indexes = []
        
        if 'Index Name' in plan_data:
            indexes.append(plan_data['Index Name'])
        
        if 'Plans' in plan_data:
            for child_plan in plan_data['Plans']:
                indexes.extend(self._extract_postgres_indexes(child_plan))
        
        return indexes

    def _identify_postgres_bottlenecks(self, plan_data: Dict) -> List[str]:
        """Identify bottlenecks from PostgreSQL plan."""
        bottlenecks = []
        
        # High cost operations
        if plan_data.get('Total Cost', 0) > 1000:
            bottlenecks.append(f"High cost operation: {plan_data.get('Node Type', 'Unknown')}")
        
        # Sequential scans on large tables
        if plan_data.get('Node Type') == 'Seq Scan' and plan_data.get('Plan Rows', 0) > 10000:
            table_name = plan_data.get('Relation Name', 'Unknown')
            bottlenecks.append(f"Large sequential scan on table {table_name}")
        
        return bottlenecks

    def _suggest_postgres_optimizations(self, plan_data: Dict) -> List[str]:
        """Suggest PostgreSQL-specific optimizations."""
        suggestions = []
        
        # Analyze for common PostgreSQL optimization opportunities
        if plan_data.get('Node Type') == 'Seq Scan':
            table_name = plan_data.get('Relation Name', 'Unknown')
            suggestions.append(f"Consider adding index to avoid sequential scan on {table_name}")
        
        return suggestions

    # Similar helper methods would be implemented for MySQL
    def _extract_mysql_scans(self, query_block: Dict) -> List[str]:
        """Extract scan operations from MySQL plan."""
        # Implementation for MySQL scan extraction
        return []

    def _extract_mysql_joins(self, query_block: Dict) -> List[str]:
        """Extract join operations from MySQL plan."""
        # Implementation for MySQL join extraction  
        return []

    def _extract_mysql_indexes(self, query_block: Dict) -> List[str]:
        """Extract index usage from MySQL plan."""
        # Implementation for MySQL index extraction
        return []

    def _identify_mysql_bottlenecks(self, query_block: Dict) -> List[str]:
        """Identify bottlenecks from MySQL plan."""
        # Implementation for MySQL bottleneck identification
        return []

    def _suggest_mysql_optimizations(self, query_block: Dict) -> List[str]:
        """Suggest MySQL-specific optimizations."""
        # Implementation for MySQL optimization suggestions
        return []
