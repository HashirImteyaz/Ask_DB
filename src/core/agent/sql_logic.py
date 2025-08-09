# agent/sql_logic.py

import os
import re
import sqlite3
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import json
import logging
from typing import Union, cast, Dict, Optional, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sqlalchemy import create_engine, inspect

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from .prompts import (
    QUERY_COMPLEXITY_ASSESSMENT_PROMPT, 
    QUERY_VALIDATION_PROMPT,
    FINAL_ANSWER_PROMPT,
    SQL_GENERATION_PROMPT,
    SCHEMA_INTROSPECTION_PROMPT,
    TABLE_INFO_PROMPT  
)
from .llm_tracker import TokenTrackingLLM, get_global_tracker

# Import the new SQL Optimizer
try:
    from ..sql.sql_optimizer import SQLQueryOptimizer, OptimizationLevel, QueryOptimizationResult
    SQL_OPTIMIZER_AVAILABLE = True
except ImportError:
    SQL_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log optimizer availability after logger is defined
if not SQL_OPTIMIZER_AVAILABLE:
    logger.warning("SQL Optimizer not available - continuing without optimization")

# Load configuration
try:
    with open("config.json", 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}

# Initialize LLM with token tracking (lazy initialization)
openai_config = CONFIG.get('openai', {})
llm = None

# Initialize SQL Optimizer (lazy initialization)
sql_optimizer = None

def get_llm():
    """Get or initialize the LLM instance"""
    global llm
    if llm is None:
        llm = TokenTrackingLLM(
            model=openai_config.get('chat_model', 'gpt-4o-mini'), 
            temperature=openai_config.get('temperature', 0)
        )
        # Register with global tracker
        get_global_tracker().register_llm("sql_logic", llm)
    return llm

def load_schema_fallback() -> str:
    """Load schema description directly when retrievers are not available."""
    try:
        schema_path = "src/config/schema_description.json"
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Create a simplified context from schema
            context_parts = []
            
            # Add business context
            if 'description' in schema_data:
                context_parts.append(f"**Business Context**: {schema_data['description']}")
            
            # Add critical query patterns
            if 'critical_query_patterns' in schema_data:
                context_parts.append("\n**Critical Query Patterns**:")
                for pattern in schema_data['critical_query_patterns']:
                    context_parts.append(f"- {pattern['name']}: {pattern['description']}")
                    if 'correct_pattern' in pattern:
                        context_parts.append(f"  Correct SQL: {pattern['correct_pattern']}")
                    if 'examples' in pattern:
                        for key, example in pattern['examples'].items():
                            context_parts.append(f"  {key}: {example}")
            
            # Add table descriptions
            if 'table_descriptions' in schema_data:
                context_parts.append("\n**Table Descriptions**:")
                for table, desc in schema_data['table_descriptions'].items():
                    context_parts.append(f"- {table}: {desc}")
            
            # Add relationships
            if 'relationships' in schema_data:
                context_parts.append("\n**Table Relationships**:")
                for rel in schema_data['relationships']:
                    context_parts.append(f"- {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}: {rel['description']}")
            
            return '\n'.join(context_parts)
    
    except Exception as e:
        logger.warning(f"Could not load schema fallback: {e}")
    
    # Basic fallback context
    return """**Database Schema**:
- Specifications table: Contains both recipes (SpecGroupCode='CUC') and ingredients (SpecGroupCode='ING')
- RecipeExplosion table: Contains recipe-ingredient relationships via CUCSpecCode and INGSpecCode

**Critical SQL Patterns**:
- List all ingredients: SELECT DISTINCT SpecDescription FROM Specifications WHERE SpecGroupCode = 'ING' LIMIT 100
- List all recipes: SELECT DISTINCT SpecDescription FROM Specifications WHERE SpecGroupCode = 'CUC' LIMIT 100
- Find ingredients of recipe: Use JOIN pattern with RecipeExplosion table"""

def get_sql_optimizer():
    """Get or initialize the SQL Optimizer instance"""
    global sql_optimizer
    if sql_optimizer is None and SQL_OPTIMIZER_AVAILABLE:
        try:
            # Create database engine for optimizer
            engine = create_engine(f"sqlite:///{DB_URL}")
            optimizer_config = CONFIG.get('sql_optimizer', {
                'optimization_level': 'intermediate',
                'enable_auto_optimization': True,
                'enable_explain_analysis': True
            })
            sql_optimizer = SQLQueryOptimizer(engine, optimizer_config)
            logger.info("SQL Optimizer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SQL Optimizer: {e}")
            sql_optimizer = None
    return sql_optimizer

# Database URL configuration
db_config = CONFIG.get('database', {})
DB_URL = os.environ.get("DB_URL") or db_config.get('default_url', "sqlite:///uploaded_data.db")
DB_URL = DB_URL.replace("sqlite:///", "")

# ==================== ENHANCED SQL EXTRACTION ====================

def assess_query_complexity(user_query: str, schema_context: str = "", chat_history: str = "") -> Dict:
    """
    Assess query complexity using the new intelligent assessment prompt.
    
    Args:
        user_query: The user's natural language query
        schema_context: Available schema information
        chat_history: Recent conversation context
    
    Returns:
        Dictionary with complexity assessment details
    """
    prompt = QUERY_COMPLEXITY_ASSESSMENT_PROMPT.format(
        user_query=user_query,
        schema_context=schema_context or "Limited schema information available",
        chat_history=chat_history or "No prior conversation"
    )
    
    try:
        response = get_llm().invoke(prompt, call_type="complexity_assessment")
        
        # Parse JSON response
        import json
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        complexity_data = json.loads(content)
        return complexity_data
        
    except Exception as e:
        logger.warning(f"Complexity assessment failed: {e}")
        # Fallback to basic assessment
        return {
            "overall_complexity": "moderate",
            "computational_complexity": "moderate", 
            "recommended_approach": "enhanced",
            "human_intervention_recommended": False,
            "reasoning": "Fallback assessment due to parsing error"
        }

def validate_sql_query(user_question: str, sql_query: str, schema_info: str, business_rules: str, expected_results: str = "") -> Dict:
    """
    Comprehensive SQL query validation using the enhanced validation prompt.
    
    Args:
        user_question: Original user question
        sql_query: Generated SQL query to validate
        schema_info: Available schema information
        business_rules: Business context and rules
        expected_results: Expected result characteristics
    
    Returns:
        Dictionary with comprehensive validation results
    """
    prompt = QUERY_VALIDATION_PROMPT.format(
        user_question=user_question,
        sql_query=sql_query,
        schema_info=schema_info,
        business_rules=business_rules,
        expected_results=expected_results or "Standard data retrieval expected"
    )
    
    try:
        response = get_llm().invoke(prompt, call_type="query_validation")
        
        # Parse JSON response
        import json
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        validation_data = json.loads(content)
        return validation_data
        
    except Exception as e:
        logger.warning(f"Query validation failed: {e}")
        # Fallback validation
        return {
            "validation_result": "conditional_pass",
            "overall_confidence": "medium",
            "total_score": 70,
            "safe_to_execute": True,
            "identified_issues": [],
            "reasoning": f"Fallback validation due to parsing error: {e}"
        }


def extract_sql_from_response(response_content: str) -> str:
    """Extract clean SQL from LLM response with robust parsing."""
    if not isinstance(response_content, str):
        return ""
    
    # Method 1: Extract from code blocks - FIXED patterns
    patterns = [
        r"```sql\s*(SELECT.*?)```",     # ```sql SELECT ... ```
        r"```\s*(SELECT.*?)```",        # ``` SELECT ... ```
        r"`(SELECT[^`]*)`"              # `SELECT ...`
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            return clean_sql_query(sql)
    
    # Method 2: Look for SQL starting with SELECT (no backticks)
    lines = response_content.split('\n')
    sql_lines = []
    capturing = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Start capturing when we see SELECT (ignore "sql SELECT")
        if re.match(r'^(?:sql\s+)?SELECT\b', line, re.IGNORECASE):
            # Remove "sql" prefix if present
            clean_line = re.sub(r'^sql\s+', '', line, flags=re.IGNORECASE)
            capturing = True
            sql_lines.append(clean_line)
        elif capturing:
            if line.startswith('**') or line.startswith('#') or 'Response:' in line:
                break
            sql_lines.append(line)
    
    if sql_lines:
        return clean_sql_query(' '.join(sql_lines))
    
    # Method 3: Fallback - clean the raw response
    cleaned = response_content.strip()
    # Remove "sql" prefix if it's there
    cleaned = re.sub(r'^sql\s+', '', cleaned, flags=re.IGNORECASE)
    return clean_sql_query(cleaned)


def clean_sql_query(sql: str) -> str:
    """Clean and validate SQL query."""
    # Remove "sql" prefix if present
    sql = re.sub(r'^sql\s+', '', sql, flags=re.IGNORECASE)

    # Remove markdown code block artifacts
    sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)  # Remove opening ```sql
    sql = re.sub(r'```', '', sql)  # Remove any remaining backticks
    sql = sql.strip()
    sql = re.sub(r'```\s*', '', sql)     # Remove ```
    
    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Clean whitespace
    sql = ' '.join(sql.split())
    
    # Ensure semicolon
    if sql and not sql.rstrip().endswith(';'):
        sql = sql.rstrip() + ';'
    
    return sql



# ==================== DATABASE-AGNOSTIC RELATIONSHIP DETECTION ====================

def detect_table_relationships() -> Dict[str, Dict]:
    """Dynamically detect relationships between tables in a database-agnostic way."""
    try:
        engine = create_engine(f"sqlite:///{DB_URL}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        relationships = {}
        
        # Get all table columns
        table_columns = {}
        for table in tables:
            table_columns[table] = [col['name'] for col in inspector.get_columns(table)]
        
        # Find potential relationships by matching column names
        for table1 in tables:
            for table2 in tables:
                if table1 == table2:
                    continue
                
                # Look for matching column names
                common_columns = set(table_columns[table1]) & set(table_columns[table2])
                
                for common_col in common_columns:
                    # Skip generic columns
                    if common_col.lower() in ['id', 'name', 'description', 'status', 'type']:
                        continue
                    
                    # If column suggests it's a key
                    if any(suffix in common_col.upper() for suffix in ['KEY', 'ID', 'NR', 'NUM']):
                        key = f"{table1}_{table2}"
                        relationships[key] = {
                            "table1": table1,
                            "table2": table2,
                            "join_condition": f"{table1}.{common_col} = {table2}.{common_col}",
                            "relationship_type": "unknown"
                        }
        
        return relationships
        
    except Exception as e:
        print(f"Warning: Could not detect relationships: {e}")
        return {}

def perform_schema_introspection(user_query: str, complexity_assessment: Dict) -> str:
    """
    Perform intelligent schema introspection based on query complexity and requirements.
    
    Args:
        user_query: The user's natural language query
        complexity_assessment: Complexity assessment data
    
    Returns:
        Schema introspection context for enhanced SQL generation
    """
    
    # Only perform detailed introspection for complex queries or when schema validation is needed
    if (complexity_assessment.get('needs_schema_check', False) or 
        complexity_assessment.get('overall_complexity') in ['complex', 'advanced'] or
        complexity_assessment.get('schema_dependency', 'low') in ['high', 'critical']):
        
        try:
            # Use the SCHEMA_INTROSPECTION_PROMPT to analyze requirements
            introspection_prompt = SCHEMA_INTROSPECTION_PROMPT + f"""

**Current Query Analysis:**
Query: "{user_query}"
Complexity: {complexity_assessment.get('overall_complexity', 'moderate')}
Schema Dependency: {complexity_assessment.get('schema_dependency', 'medium')}
Recommended Approach: {complexity_assessment.get('recommended_approach', 'enhanced')}

Based on this query analysis, provide a focused schema validation plan that addresses the specific requirements and complexity level identified.
"""
            
            response = get_llm().invoke(introspection_prompt, call_type="schema_introspection")
            introspection_plan = response.content.strip()
            
            # Execute basic schema discovery queries
            schema_info = execute_schema_discovery_queries()
            
            # Combine plan with actual schema info
            enhanced_context = f"""
**Schema Introspection Results (Applied for {complexity_assessment.get('overall_complexity', 'moderate')} query):**

{introspection_plan}

**Discovered Schema Information:**
{schema_info}

**Validation Requirements:**
- Schema dependency level: {complexity_assessment.get('schema_dependency', 'medium')}
- All table and column references must be verified against the above schema
- Use conservative JOIN patterns when relationships are uncertain
- Apply appropriate data type constraints in WHERE conditions
- Consider fallback strategies if complex relationships fail
"""
            
            logger.info(f"Schema introspection completed for {complexity_assessment.get('overall_complexity')} query")
            return enhanced_context
            
        except Exception as e:
            logger.warning(f"Schema introspection failed: {e}")
            return ""
    
    return ""

def execute_schema_discovery_queries() -> str:
    """Execute actual schema discovery SQL queries."""
    try:
        with sqlite3.connect(DB_URL) as conn:
            # Get table list
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            tables_df = pd.read_sql_query(tables_query, conn)
            
            schema_info = "**Available Tables:**\n"
            for table in tables_df['name']:
                schema_info += f"- {table}\n"
                
                # Get column info for each table
                columns_query = f"PRAGMA table_info({table});"
                columns_df = pd.read_sql_query(columns_query, conn)
                
                schema_info += f"  Columns: {', '.join(columns_df['name'].tolist())}\n"
                
                # Sample data for pattern recognition (limit to 2 rows)
                try:
                    sample_query = f"SELECT * FROM {table} LIMIT 2;"
                    sample_df = pd.read_sql_query(sample_query, conn)
                    if not sample_df.empty:
                        schema_info += f"  Sample values: {sample_df.iloc[0].to_dict()}\n"
                except Exception:
                    pass
                    
                schema_info += "\n"
            
            return schema_info
            
    except Exception as e:
        logger.warning(f"Schema discovery execution failed: {e}")
        return "Schema discovery not available"
# ==================== ENHANCED SQL GENERATION ====================

def preprocess_search_terms(user_query: str) -> str:
    """Preprocess user query to handle compound search terms."""
    import re
    
    # Handle hyphenated terms by converting to space-separated for better context
    processed_query = user_query
    
    # Find hyphenated terms (but not SQL operators or common words)
    hyphenated_pattern = r'\b([a-zA-Z]+)-([a-zA-Z]+)\b'
    hyphenated_matches = re.findall(hyphenated_pattern, processed_query)
    
    for match in hyphenated_matches:
        original_term = f"{match[0]}-{match[1]}"
        # Skip common hyphenated words that shouldn't be split
        skip_terms = ['well-known', 'state-of-the-art', 'real-time', 'user-friendly']
        if original_term.lower() not in skip_terms:
            spaced_term = f"{match[0]} {match[1]}"
            processed_query = processed_query.replace(original_term, spaced_term)
    
    return processed_query

def analyze_table_metadata(table_name: str, sample_data: pd.DataFrame) -> Dict:
    """
    Analyze table metadata using the TABLE_INFO_PROMPT.
    
    Args:
        table_name: Name of the table
        sample_data: Sample data from the table
    
    Returns:
        Dictionary with enhanced table metadata
    """
    try:
        # Convert sample data to string representation
        table_str = sample_data.to_string(max_rows=5) if not sample_data.empty else "No sample data available"
        
        # Use the TABLE_INFO_PROMPT
        prompt = TABLE_INFO_PROMPT.format(
            table_name=table_name,
            table_str=table_str
        )
        
        response = get_llm().invoke(prompt, call_type="table_analysis")
        
        # Parse JSON response
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
        
    except Exception as e:
        logger.warning(f"Table metadata analysis failed: {e}")
        return {
            "table_name": table_name,
            "table_summary": "Basic table information",
            "entity_type": "unknown",
            "primary_purpose": "Data storage"
        }
def generate_sql_with_context(user_query: str, rag_context: str, conversation_context: str = "") -> str:
    """
    Generate SQL with enhanced context including complexity assessment, validation, and schema introspection.
    """
    
    # Use fallback if rag_context is empty or insufficient
    if not rag_context or len(rag_context.strip()) < 50:
        logger.info("RAG context insufficient, using schema fallback")
        rag_context = load_schema_fallback()
    
    # Step 1: Assess query complexity
    complexity_assessment = assess_query_complexity(
        user_query=user_query,
        schema_context=rag_context,
        chat_history=conversation_context
    )
    
    logger.info(f"Query complexity assessed as: {complexity_assessment.get('overall_complexity', 'unknown')}")
    
    # Step 2: NEW - Perform schema introspection if needed
    schema_introspection_context = perform_schema_introspection(user_query, complexity_assessment)
    
    # Step 3: Check if human intervention is recommended
    if complexity_assessment.get('human_intervention_recommended', False):
        logger.warning("Complex query detected - may require human intervention")
    
    # Step 4: Preprocess search terms
    processed_query = preprocess_search_terms(user_query)
    
    # Step 5: Get table relationships
    relationships = detect_table_relationships()
    
    # Step 6: Build enhanced context based on complexity
    join_hints = ""
    if relationships and complexity_assessment.get('computational_complexity') in ['complex', 'advanced']:
        join_hints = "\n**Available Table Relationships:**\n"
        for rel_key, rel_info in relationships.items():
            join_hints += f"- {rel_info['join_condition']}\n"
    
    # Add conversation context if available
    context_block = ""
    if conversation_context:
        context_block = f"\n**Previous Context:**\n{conversation_context[:500]}...\n"
    
    # Add complexity guidance
    complexity_guidance = f"\n**Query Complexity Guidance:**\n"
    complexity_guidance += f"- Complexity Level: {complexity_assessment.get('overall_complexity', 'moderate')}\n"
    complexity_guidance += f"- Recommended Approach: {complexity_assessment.get('recommended_approach', 'enhanced')}\n"
    
    if complexity_assessment.get('optimization_priorities'):
        complexity_guidance += f"- Optimization Priorities: {', '.join(complexity_assessment['optimization_priorities'])}\n"
    
    # ENHANCED: Combine all contexts including schema introspection
    enhanced_context = f"{rag_context}{schema_introspection_context}{join_hints}{complexity_guidance}"
    
    # Step 7: Generate SQL with enhanced prompt
    prompt = SQL_GENERATION_PROMPT.format(
        business_rules=enhanced_context, 
        user_query=processed_query, 
        history=context_block
    )
    
    response: BaseMessage = get_llm().invoke(prompt, call_type="sql_generation_enhanced")
    sql = extract_sql_from_response(response.content)
    
    # Step 8: Validate generated SQL if complexity warrants it
    if sql and complexity_assessment.get('overall_complexity') in ['complex', 'advanced']:
        validation_result = validate_sql_query(
            user_question=user_query,
            sql_query=sql,
            schema_info=enhanced_context,  # Use enhanced context for validation
            business_rules=enhanced_context
        )
        
        # Log validation results
        logger.info(f"SQL validation score: {validation_result.get('total_score', 'unknown')}")
        
        # If validation fails badly, try a simpler approach
        if (validation_result.get('total_score', 100) < 50 or 
            not validation_result.get('safe_to_execute', True)):
            
            logger.warning("Generated SQL failed validation, attempting simpler approach")
            
            # Regenerate with simpler context
            simple_prompt = SQL_GENERATION_PROMPT.format(
                business_rules=rag_context,  # Use basic context only
                user_query=processed_query,
                history=""  # Remove conversation context
            )
            
            simple_response = get_llm().invoke(simple_prompt, call_type="sql_generation_simple")
            sql = extract_sql_from_response(simple_response.content)
    
    if sql:
        print(f"DEBUG: Generated SQL with enhanced intelligence and schema introspection:\n{sql}")
        print(f"DEBUG: Query complexity: {complexity_assessment.get('overall_complexity')}")
        print(f"DEBUG: Schema introspection applied: {len(schema_introspection_context) > 0}")
        return sql
    else:
        print("Warning: Could not extract SQL from LLM response")
        return ""

def generate_sql(user_query: str, rag_context: str) -> str:
    """Main SQL generation function (maintains compatibility)."""
    return generate_sql_with_context(user_query, rag_context)

def execute_sql(query: str, page_number: int = 1, page_size: int = 1000, return_total_count: bool = False, enable_optimization: bool = True) -> Union[pd.DataFrame, str, Dict]:
    # --- Aggregation/Grouping Post-Processing ---
    original_query = query  # Ensure original_query is initialized before use
    def needs_aggregation(user_query: str) -> bool:
        # Simple intent detection for aggregation/grouping
        agg_keywords = ["number of", "count", "sum", "average", "per", "each", "group", "total", "how many", "contribute", "distribution"]
        uq = user_query.lower()
        return any(kw in uq for kw in agg_keywords)

    def is_aggregation_sql(sql: str) -> bool:
        # Check for GROUP BY, COUNT, SUM, AVG
        sql_lc = sql.lower()
        return any(x in sql_lc for x in ["group by", "count(", "sum(", "avg("])

    # If user intent needs aggregation, but SQL does not have it, warn or retry
    if needs_aggregation(original_query) and not is_aggregation_sql(query):
        logger.warning("User intent requires aggregation/grouping, but SQL does not contain GROUP BY/COUNT/SUM/AVG. Query may be incorrect.")
    """
    Executes a SQL query against the SQLite database with enhanced error handling, 
    pagination, and intelligent optimization.
    
    Args:
        query: SQL query to execute
        page_number: Page number (1-based)
        page_size: Number of rows per page
        return_total_count: Whether to return total count information
        enable_optimization: Whether to apply SQL optimization
    
    Returns:
        DataFrame, error string, or dict with pagination info
    """
    if not query:
        return "Cannot execute an empty query."
    
    original_query = query
    optimization_result = None
    
    # Apply SQL optimization if enabled and available
    if enable_optimization and SQL_OPTIMIZER_AVAILABLE:
        optimizer = get_sql_optimizer()
        if optimizer:
            try:
                optimizer_config = CONFIG.get('sql_optimizer', {})
                optimization_level = OptimizationLevel(optimizer_config.get('optimization_level', 'intermediate'))
                
                logger.info(f"Analyzing query optimization opportunities...")
                optimization_result = optimizer.optimize_query(query, optimization_level)
                
                # Use optimized query if safe and shows significant improvement
                if (optimization_result.safe_to_auto_apply and 
                    optimization_result.confidence > 0.6 and
                    optimization_result.improvement_summary.get('cost_improvement_percent', 0) > 10):
                    
                    query = optimization_result.optimized_query
                    logger.info(f"Applied automatic optimization - estimated {optimization_result.improvement_summary.get('cost_improvement_percent', 0):.1f}% improvement")
                else:
                    logger.info(f"Optimization analysis completed - keeping original query (confidence: {optimization_result.confidence:.2f})")
                    
            except Exception as e:
                logger.warning(f"Query optimization failed: {e} - using original query")

    try:
        with sqlite3.connect(DB_URL) as conn:
            conn: sqlite3.Connection
            
            # DEBUG: Print database path and query for troubleshooting
            print(f"DEBUG: Database path: {DB_URL}")
            print(f"DEBUG: Executing query: {query}")
            
            # Handle pagination
            if page_number > 1 or page_size < 10000:  # Apply pagination for reasonable page sizes
                # First get total count if requested
                total_rows = None
                if return_total_count:
                    # Clean the original query by removing trailing semicolon
                    clean_query = original_query.rstrip(';').strip()
                    count_query = f"SELECT COUNT(*) as total_count FROM ({clean_query})"
                    try:
                        count_df = pd.read_sql_query(count_query, conn)
                        total_rows = count_df['total_count'].iloc[0]
                    except Exception as e:
                        logger.warning(f"Failed to get total count: {e}")
                
                # Apply pagination to the (potentially optimized) query
                # Check if query already has LIMIT clause to avoid double limits
                offset = (page_number - 1) * page_size
                query_upper = query.upper().strip()
                if 'LIMIT' in query_upper:
                    # Query already has LIMIT clause, don't add pagination to avoid SQL error
                    print(f"DEBUG: Query already has LIMIT clause, using as-is: {query}")
                    paginated_query = query
                else:
                    paginated_query = f"{query} LIMIT {page_size} OFFSET {offset}"
                
                df: pd.DataFrame = pd.read_sql_query(paginated_query, conn)
                
                if return_total_count:
                    result = {
                        'data': df,
                        'page': page_number,
                        'page_size': page_size,
                        'total_rows': total_rows,
                        'has_more': len(df) == page_size,
                        'total_pages': (total_rows + page_size - 1) // page_size if total_rows else None
                    }
                    
                    # Add optimization information if available
                    if optimization_result:
                        result['optimization_info'] = {
                            'optimization_applied': query != original_query,
                            'confidence': optimization_result.confidence,
                            'estimated_improvement_percent': optimization_result.improvement_summary.get('cost_improvement_percent', 0),
                            'suggestions_count': len(optimization_result.suggestions),
                            'bottlenecks_identified': len(optimization_result.original_plan.bottlenecks),
                            'safe_to_auto_apply': optimization_result.safe_to_auto_apply
                        }
                    
                    return result
                else:
                    if df.empty and page_number > 1:
                        return f"No data found on page {page_number}. This page may be beyond the available data."
                    elif df.empty:
                        return "No data found matching your criteria."
                    return df
            else:
                # Execute without pagination for small datasets
                df: pd.DataFrame = pd.read_sql_query(query, conn)
                if df.empty:
                    return "No data found matching your criteria."
                
                # Check if result is very large and warn
                if len(df) > 10000:
                    logger.warning(f"Large result set returned: {len(df)} rows")
                    # Truncate to prevent memory issues
                    if len(df) > 50000:
                        df = df.head(50000)
                        optimization_warning = ""
                        if optimization_result and not optimization_result.safe_to_auto_apply:
                            optimization_warning = f" Optimization suggestions available but not auto-applied - {len(optimization_result.suggestions)} suggestions found."
                        return f"Result truncated to 50,000 rows out of {len(df)} total. Use pagination for full results.{optimization_warning}"
                
                # Add optimization info to successful results if available
                if optimization_result and hasattr(df, 'attrs'):
                    df.attrs['optimization_info'] = {
                        'optimization_applied': query != original_query,
                        'confidence': optimization_result.confidence,
                        'estimated_improvement_percent': optimization_result.improvement_summary.get('cost_improvement_percent', 0),
                        'suggestions_count': len(optimization_result.suggestions)
                    }
                
                return df
                
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        print(f"DEBUG: SQLite OperationalError: {e}")
        print(f"DEBUG: Query that failed: {query}")
        
        # Provide specific guidance based on common errors
        if "no such table" in error_msg:
            return f"SQL Error: Table not found. Please check if the data has been uploaded correctly. Error: {e}"
        elif "no such column" in error_msg:
            return f"SQL Error: Column not found. Please verify the column names in your query. Error: {e}"
        elif "syntax error" in error_msg:
            return f"SQL Error: Syntax error in the query. The query may need to be reformulated. Error: {e}"
        else:
            return f"SQL Execution Error: {e}\nQuery: {query}\n\nPlease try rephrasing your question or provide more specific details about what you're looking for."
    except Exception as e:
        print(f"DEBUG: Unexpected exception: {e}")
        print(f"DEBUG: Query that failed: {query}")
        return f"Unexpected Error: {e}\nQuery: {query}\n\nPlease try rephrasing your question or contact support if the issue persists."

def generate_error_clarification(user_query: str, sql_query: str, error_msg: str) -> str:
    """Generate clarification questions when SQL fails."""
    clarification_prompt = f"""
The user asked: "{user_query}"
Generated SQL: {sql_query}
Error: {error_msg}

Generate a helpful clarification question to ask the user. Focus on:
1. What specific information they're looking for
2. Which tables/columns they might be interested in
3. Any ambiguity in their request

Return only the clarification question:
"""
    
    try:
        response = get_llm().invoke(clarification_prompt, call_type="error_clarification")
        return response.content
    except Exception:
        return "I encountered an error processing your request. Could you please rephrase your question with more specific details about what you're looking for?"

def analyze_sql_optimization(sql_query: str) -> Optional[Dict]:
    """
    Analyze SQL query for optimization opportunities without executing it.
    
    Args:
        sql_query: SQL query to analyze
        
    Returns:
        Dictionary with optimization analysis or None if optimizer not available
    """
    if not SQL_OPTIMIZER_AVAILABLE or not sql_query:
        return None
    
    optimizer = get_sql_optimizer()
    if not optimizer:
        return None
    
    try:
        optimization_result = optimizer.optimize_query(sql_query, OptimizationLevel.INTERMEDIATE)
        
        return {
            'original_query': sql_query,
            'optimized_query': optimization_result.optimized_query,
            'optimization_applied': sql_query != optimization_result.optimized_query,
            'confidence': optimization_result.confidence,
            'safe_to_auto_apply': optimization_result.safe_to_auto_apply,
            'improvement_summary': optimization_result.improvement_summary,
            'suggestions': [
                {
                    'type': suggestion.type,
                    'priority': suggestion.priority,
                    'description': suggestion.description,
                    'estimated_improvement_percent': suggestion.estimated_improvement_percent,
                    'reasoning': suggestion.reasoning
                }
                for suggestion in optimization_result.suggestions
            ],
            'execution_plan_analysis': {
                'original_cost': optimization_result.original_plan.estimated_cost,
                'optimized_cost': optimization_result.optimized_plan.estimated_cost,
                'bottlenecks': optimization_result.original_plan.bottlenecks,
                'optimization_opportunities': optimization_result.original_plan.optimization_opportunities,
                'index_usage': optimization_result.original_plan.index_usage,
                'scan_operations': optimization_result.original_plan.scan_operations
            }
        }
        
    except Exception as e:
        logger.warning(f"SQL optimization analysis failed: {e}")
        return None

def get_optimization_recommendations(sql_query: str) -> List[Dict]:
    """
    Get specific optimization recommendations for a SQL query.
    
    Args:
        sql_query: SQL query to analyze
        
    Returns:
        List of optimization recommendations
    """
    analysis = analyze_sql_optimization(sql_query)
    
    if not analysis:
        return []
    
    recommendations = []
    
    # Add suggestions as recommendations
    for suggestion in analysis.get('suggestions', []):
        recommendations.append({
            'category': 'optimization',
            'priority': suggestion['priority'],
            'title': suggestion['type'].replace('_', ' ').title(),
            'description': suggestion['description'],
            'reasoning': suggestion['reasoning'],
            'estimated_benefit': f"{suggestion['estimated_improvement_percent']:.1f}% improvement",
            'implementation_complexity': 'low' if suggestion['priority'] == 'low' else 'medium'
        })
    
    # Add execution plan insights
    exec_plan = analysis.get('execution_plan_analysis', {})
    
    if exec_plan.get('bottlenecks'):
        for bottleneck in exec_plan['bottlenecks']:
            recommendations.append({
                'category': 'performance_issue',
                'priority': 'high',
                'title': 'Performance Bottleneck Detected',
                'description': bottleneck,
                'reasoning': 'This operation may cause significant performance degradation',
                'estimated_benefit': 'High performance improvement possible',
                'implementation_complexity': 'medium'
            })
    
    if exec_plan.get('optimization_opportunities'):
        for opportunity in exec_plan['optimization_opportunities']:
            recommendations.append({
                'category': 'optimization_opportunity',
                'priority': 'medium',
                'title': 'Optimization Opportunity',
                'description': opportunity,
                'reasoning': 'This optimization could improve query performance',
                'estimated_benefit': 'Moderate performance improvement',
                'implementation_complexity': 'low'
            })
    
    return recommendations[:10]  # Limit to top 10 recommendations

def retry_sql_generation(user_query: str, rag_context: str, previous_sql: str, error_msg: str, attempt: int) -> str:
    """
    Enhanced retry generation with complexity assessment and validation.
    """
    
    # Assess complexity for retry strategy
    complexity_assessment = assess_query_complexity(
        user_query=user_query,
        schema_context=rag_context
    )
    
    # Build retry prompt with enhanced context
    retry_prompt = f"""
The previous SQL query failed. Generate a corrected SQL query using enhanced analysis.

**Query Analysis:**
- Original Question: {user_query}
- Complexity Level: {complexity_assessment.get('overall_complexity', 'moderate')}
- Failed SQL: {previous_sql}
- Error Message: {error_msg}
- Retry Attempt: {attempt}/3

**Database Context:** 
{rag_context}

**Enhanced Retry Strategy:**
1. **Error Analysis**: Examine the specific error and identify root cause
2. **Complexity Consideration**: Use {complexity_assessment.get('recommended_approach', 'standard')} approach
3. **Simplification**: If attempt > 1, prefer simpler query patterns
4. **Validation**: Ensure all table/column references are valid
5. **Fallback Logic**: Use conservative assumptions about schema

**Critical Requirements:**
- Fix the specific error mentioned
- Use only verified table and column names from context
- Apply appropriate complexity level for this retry attempt
- Include proper error prevention (NULL handling, case sensitivity)
- Add reasonable LIMIT clauses

**Retry Approach for Attempt {attempt}:**
{_get_retry_strategy(attempt, complexity_assessment)}

Return only the corrected SQL query:
"""
    
    try:
        response = get_llm().invoke(retry_prompt, call_type=f"sql_retry_attempt_{attempt}")
        sql = extract_sql_from_response(response.content)
        
        # Enhanced validation for retry using QUERY_VALIDATION_PROMPT
        if sql and attempt <= 2:  # Only validate if not final attempt
            validation = validate_sql_query(
                user_question=user_query,
                sql_query=sql,
                schema_info=rag_context,
                business_rules=rag_context,  # Use rag_context as business rules
                expected_results=f"Retry attempt {attempt} - Error correction focused"
            )
            
            # Log validation results for retry
            logger.info(f"Retry attempt {attempt} validation score: {validation.get('total_score', 'unknown')}")
            
            if not validation.get('safe_to_execute', True):
                logger.warning(f"Retry attempt {attempt} SQL failed validation - proceeding with caution")
        
        print(f"DEBUG: Enhanced retry attempt {attempt} generated SQL: {sql}")
        return sql
    except Exception as e:
        print(f"Warning: Failed to generate enhanced retry SQL: {e}")
        return ""
    

def _get_retry_strategy(attempt: int, complexity_assessment: Dict) -> str:
    """Generate retry strategy based on attempt number and complexity."""
    
    if attempt == 1:
        return """
**First Retry - Error Correction Focus:**
- Analyze exact error message and fix specific issue
- Maintain original query complexity and intent
- Apply defensive coding practices (NULL checks, proper JOINs)
"""
    elif attempt == 2:
        return """
**Second Retry - Simplification Strategy:**
- Reduce query complexity if possible
- Use simpler JOIN patterns or single-table approach
- Focus on core requirement rather than optimization
- Add more conservative error handling
"""
    else:  # attempt == 3
        return """
**Final Retry - Maximum Simplification:**
- Use simplest possible query structure
- Single table queries preferred
- Basic filtering only
- Conservative approach to ensure execution success
"""


def generate_sql_with_intelligence_pipeline(user_query: str, rag_context: str, conversation_context: str = "") -> Dict:
    """
    Complete intelligent SQL generation pipeline with assessment, generation, and validation.
    
    Returns:
        Dictionary with SQL, complexity assessment, validation results, and metadata
    """
    
    # Step 1: Complexity Assessment  
    complexity = assess_query_complexity(user_query, rag_context, conversation_context)
    
    # Step 2: SQL Generation
    sql = generate_sql_with_context(user_query, rag_context, conversation_context)
    
    # Step 3: Validation (for complex queries)
    validation = None
    if sql and complexity.get('overall_complexity') in ['complex', 'advanced']:
        validation = validate_sql_query(user_query, sql, rag_context, rag_context)
    
    return {
        'sql_query': sql,
        'complexity_assessment': complexity,
        'validation_results': validation,
        'recommended_approach': complexity.get('recommended_approach', 'enhanced'),
        'confidence_score': validation.get('total_score', 75) if validation else 75,
        'safe_to_execute': validation.get('safe_to_execute', True) if validation else True,
        'metadata': {
            'generation_method': 'intelligent_pipeline',
            'complexity_level': complexity.get('overall_complexity', 'moderate'),
            'human_intervention_recommended': complexity.get('human_intervention_recommended', False)
        }
    }
def generate_final_answer(user_question: str, result: Union[pd.DataFrame, str, None], complexity_level: str = "moderate") -> str:
    """
    Generates a comprehensive answer using the enhanced FINAL_ANSWER_PROMPT.
    
    Args:
        user_question: The original user question
        result: SQL execution result
        complexity_level: Assessed complexity level for response tailoring
    
    Returns:
        Comprehensive analysis of the results
    """
    if isinstance(result, str):
        return result
    if result is None:
        return "No data found."
    
    # Convert DataFrame to text representation for the LLM
    if isinstance(result, pd.DataFrame) and not result.empty:
        # Create a readable table representation with size limits
        if len(result) > 100:
            # For large results, provide summary + sample
            table_text = f"Dataset Summary: {len(result)} total rows, {len(result.columns)} columns\n\n"
            table_text += f"Column Names: {', '.join(result.columns)}\n\n"
            table_text += "Sample Data (first 20 rows):\n"
            table_text += result.head(20).to_string(index=False, max_rows=20)
            table_text += f"\n\n[Additional {len(result)-20} rows not shown for brevity]"
        else:
            table_text = result.to_string(index=False, max_rows=50)
        
        # Use enhanced FINAL_ANSWER_PROMPT
        prompt = FINAL_ANSWER_PROMPT.format(
            user_question=user_question,
            table_text=table_text
        )
        
        try:
            response = get_llm().invoke(prompt, call_type="final_answer_enhanced")
            return response.content
        except Exception as e:
            print(f"Warning: Failed to generate enhanced final answer: {e}")
            # Fallback to simple response
            return _generate_simple_text_response(result, user_question)
    
    # Fallback for other cases
    return _generate_simple_text_response(result, user_question)

def _generate_simple_text_response(df: pd.DataFrame, user_question: str) -> str:
    """Generate a simple text response without HTML formatting for history storage."""
    if isinstance(df, str):
        return df
        
    # Single value result (like COUNT queries)
    if len(df) == 1 and len(df.columns) == 1:
        value = df.iloc[0, 0]
        if isinstance(value, (int, float)):
            return f"**{value:,}**"
        else:
            return f"**{value}**"
    
    # Small table - show as markdown
    if len(df) <= 20:
        try:
            return df.to_markdown(index=False)
        except:
            return df.to_string(index=False)
    
    # Large table - show first few rows + summary
    else:
        try:
            preview = df.head(10).to_markdown(index=False)
            return f"{preview}\n\n*({len(df)} total rows)*"
        except:
            return f"{df.head(10).to_string(index=False)}\n\n*({len(df)} total rows)*"

def format_response_for_ui(plain_text_response: str, result: Union[pd.DataFrame, str, None], user_question: str) -> str:
    """Convert plain text response to HTML format for UI display."""
    if isinstance(result, pd.DataFrame) and not result.empty:
        from .response_formatter import format_dataframe_response
        return format_dataframe_response(result, user_question)
    else:
        # For non-DataFrame results, return the plain text (it might contain markdown)
        return plain_text_response

def determine_chart_type(df: pd.DataFrame, user_query: str) -> str:
    """Determine the best chart type based on data and user query."""
    user_query_lower = user_query.lower()
    
    # User explicitly requested chart type
    if any(word in user_query_lower for word in ['pie chart', 'pie graph', 'pie']):
        return 'pie'
    elif any(word in user_query_lower for word in ['line chart', 'line graph', 'trend', 'over time', 'timeline']):
        return 'line'
    elif any(word in user_query_lower for word in ['bar chart', 'bar graph', 'compare', 'comparison']):
        return 'bar'
    
    # Auto-determine based on data characteristics
    if df.shape[0] <= 10 and df.shape[1] == 2:
        # Small dataset with 2 columns - good for pie chart
        x_col, y_col = df.columns[0], df.columns[1]
        if pd.api.types.is_numeric_dtype(df[y_col]):
            return 'pie'
    
    # Check if x-axis looks like time/date
    x_col = df.columns[0]
    if any(word in x_col.lower() for word in ['date', 'time', 'year', 'month', 'day']):
        return 'line'
    
    # Default to bar chart
    return 'bar'

def generate_graph(df: pd.DataFrame, user_query: str = "") -> Union[str, None]:
    """Generates appropriate chart (bar, line, or pie) based on data and user query."""
    if df.empty or df.shape[1] < 2:
        return None
    
    try:
        # Determine chart type
        chart_type = determine_chart_type(df, user_query)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_col, y_col = df.columns[0], df.columns[1]
        
        # Ensure y column is numeric
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return None
        
        # Limit data points for readability
        if len(df) > 20:
            df_plot = df.head(20)
            title_suffix = f" (Top 20 of {len(df)})"
        else:
            df_plot = df
            title_suffix = ""
        
        if chart_type == 'pie':
            # Pie chart
            colors = plt.cm.Set3(range(len(df_plot)))
            wedges, texts, autotexts = ax.pie(df_plot[y_col], labels=df_plot[x_col], 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title(f'{y_col} Distribution by {x_col}{title_suffix}', fontsize=16, weight='bold')
            
        elif chart_type == 'line':
            # Line chart
            ax.plot(df_plot[x_col], df_plot[y_col], marker='o', linewidth=2, markersize=6, color='#667eea')
            ax.set_title(f'{y_col} Trend by {x_col}{title_suffix}', fontsize=16, weight='bold')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
        else:  # bar chart (default)
            bars = ax.bar(df_plot[x_col], df_plot[y_col], color='#667eea', alpha=0.8)
            ax.set_title(f'{y_col} by {x_col}{title_suffix}', fontsize=16, weight='bold')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
        
    except Exception as e:
        print(f"Failed to generate graph: {e}")
        return None




