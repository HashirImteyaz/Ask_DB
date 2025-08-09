
import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import inspect, text, MetaData, Table
from sqlalchemy.engine import Engine

# LlamaIndex imports with fallback
try:
    from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    LLAMA_INDEX_AVAILABLE = True
    
    # Configure Settings with OpenAI credentials
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if os.getenv('OPENAI_API_KEY'):
        Settings.embed_model = OpenAIEmbedding(api_key=os.getenv('OPENAI_API_KEY'))
        Settings.llm = LlamaOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")
        print("✅ LlamaIndex configured with OpenAI credentials")
    else:
        print("⚠️ OpenAI API key not found, LlamaIndex will use defaults")
        
except ImportError:
    # Try alternative import paths for different versions
    try:
        from llama_index import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage
        from llama_index.retrievers import BaseRetriever
        from llama_index.node_parser import SemanticSplitterNodeParser, SentenceSplitter
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        LLAMA_INDEX_AVAILABLE = True
        
        # Configure Settings with OpenAI credentials
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv('OPENAI_API_KEY'):
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv('OPENAI_API_KEY'))
            Settings.llm = LlamaOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")
            print("✅ LlamaIndex (alternative path) configured with OpenAI credentials")
        
    except ImportError:
        LLAMA_INDEX_AVAILABLE = False
        Settings = None
        VectorStoreIndex = None
        Document = None
        StorageContext = None
        load_index_from_storage = None
        BaseRetriever = None
        SemanticSplitterNodeParser = None
        SentenceSplitter = None
        OpenAIEmbedding = None
        LlamaOpenAI = None
        print("❌ LlamaIndex import failed - falling back to basic functionality")

from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np

# Scikit-learn imports with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    cosine_similarity = None

class RetrievalLayer(Enum):
    BUSINESS_RULES = "business_rules"
    SCHEMA = "schema" 
    SAMPLES = "samples"
    STATISTICS = "statistics"
    RELATIONSHIPS = "relationships"
    QUERY_PATTERNS = "query_patterns"

@dataclass
class RetrievalContext:
    query: str
    classification: Any  # QueryClassification
    table_context: Dict
    token_budget: int

class HierarchicalRAGSystem:
    """Advanced RAG system with intelligent context management."""
    
    def __init__(self, engine: Engine, config_path: str = "config.json"):
        self.engine = engine
        self.config = self._load_config(config_path)
        self.rag_config = self.config.get('rag_system', {})
        
        # Initialize storage
        self.storage_dir = Path(self.rag_config.get('storage_dir', 'rag_storage'))
        self.storage_dir.mkdir(exist_ok=True)
        
        # Cache for retrievers and contexts
        self.retrievers: Dict[RetrievalLayer, Optional[BaseRetriever]] = {}
        self.context_cache: Dict[str, str] = {}
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        # Database statistics cache
        self.db_stats: Dict[str, Dict] = {}
        
        # Initialize components
        self._setup_embedding_model()
        self._initialize_database_stats()

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _setup_embedding_model(self):
        """Setup optimized embedding model based on query classification."""
        if not LLAMA_INDEX_AVAILABLE or OpenAIEmbedding is None:
            self.embedding_models = {}
            return
            
        try:
            self.embedding_models = {
                "general": OpenAIEmbedding(
                    model="text-embedding-3-small",
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                ),
                "domain": OpenAIEmbedding(
                    model="text-embedding-3-large", 
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                ),
                "technical": OpenAIEmbedding(
                    model="text-embedding-3-large",
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                )
            }
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI embeddings: {e}")
            self.embedding_models = {}
        Settings.embed_model = self.embedding_models["general"]

    def _initialize_database_stats(self):
        """Initialize and cache database statistics."""
        try:
            inspector = inspect(self.engine)
            with self.engine.connect() as conn:
                for table_name in inspector.get_table_names():
                    # Get row count
                    count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                    
                    # Get column info with statistics
                    columns = inspector.get_columns(table_name)
                    column_stats = {}
                    
                    for col in columns:
                        col_name = col['name']
                        col_type = str(col['type']).lower()
                        
                        # Get basic statistics for numeric columns
                        if any(t in col_type for t in ['int', 'float', 'numeric', 'decimal']):
                            stats_query = text(f'''
                                SELECT 
                                    MIN("{col_name}") as min_val,
                                    MAX("{col_name}") as max_val,
                                    AVG("{col_name}") as avg_val,
                                    COUNT(DISTINCT "{col_name}") as distinct_count
                                FROM "{table_name}"
                                WHERE "{col_name}" IS NOT NULL
                            ''')
                            stats = conn.execute(stats_query).fetchone()
                            column_stats[col_name] = {
                                'type': col_type,
                                'min': stats[0] if stats else None,
                                'max': stats[1] if stats else None,
                                'avg': stats[2] if stats else None,
                                'distinct_count': stats[3] if stats else None
                            }
                        else:
                            # For text columns, get distinct count and sample values
                            distinct_query = text(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}" WHERE "{col_name}" IS NOT NULL')
                            distinct_count = conn.execute(distinct_query).scalar()
                            
                            sample_query = text(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT 5')
                            samples = [row[0] for row in conn.execute(sample_query).fetchall()]
                            
                            column_stats[col_name] = {
                                'type': col_type,
                                'distinct_count': distinct_count,
                                'samples': samples
                            }
                    
                    self.db_stats[table_name] = {
                        'row_count': count_result,
                        'columns': column_stats
                    }
                    
        except Exception as e:
            print(f"Warning: Could not initialize database statistics: {e}")

    def build_enhanced_retrievers(self, schema_data: Optional[Dict] = None) -> Dict[RetrievalLayer, Optional[BaseRetriever]]:
        """Build comprehensive retriever system with caching."""
        
        # If LlamaIndex is not available, return empty retrievers
        if not LLAMA_INDEX_AVAILABLE:
            print("LlamaIndex not available, returning empty retrievers")
            return {
                RetrievalLayer.BUSINESS_RULES: None,
                RetrievalLayer.SCHEMA: None,
                RetrievalLayer.SAMPLES: None,
                RetrievalLayer.STATISTICS: None,
                RetrievalLayer.RELATIONSHIPS: None,
                RetrievalLayer.QUERY_PATTERNS: None
            }
        
        # Check if we can load from cache
        cache_key = self._get_cache_key(schema_data)
        cached_path = self.storage_dir / f"retrievers_{cache_key}"
        
        if cached_path.exists() and self.rag_config.get('use_cache', True):
            return self._load_cached_retrievers(cached_path)
        
        print("Building enhanced retriever system...")
        
        # Build each retriever layer
        self.retrievers[RetrievalLayer.BUSINESS_RULES] = self._build_business_rules_retriever(schema_data)
        self.retrievers[RetrievalLayer.SCHEMA] = self._build_enhanced_schema_retriever(schema_data)
        self.retrievers[RetrievalLayer.SAMPLES] = self._build_smart_samples_retriever(schema_data)
        self.retrievers[RetrievalLayer.STATISTICS] = self._build_statistics_retriever()
        self.retrievers[RetrievalLayer.RELATIONSHIPS] = self._build_relationships_retriever()
        self.retrievers[RetrievalLayer.QUERY_PATTERNS] = self._build_query_patterns_retriever()
        
        # Cache the retrievers
        if self.rag_config.get('use_cache', True):
            self._cache_retrievers(cached_path)
        
        return self.retrievers

    def _build_business_rules_retriever(self, schema_data: Dict) -> Optional[BaseRetriever]:
        """Enhanced business rules with semantic chunking."""
        if not schema_data or not LLAMA_INDEX_AVAILABLE:
            return None
        
        documents = []
        
        # Process business rules with enhanced context
        if "description" in schema_data:
            doc = Document(
                text=f"Business Context Overview: {schema_data['description']}", 
                metadata={"source": "rules", "type": "context", "priority": "high"}
            )
            documents.append(doc)

        # Enhanced relationship processing
        for rel in schema_data.get("relationships", []):
            relationship_text = (
                f"Database Relationship: {rel['from_table']}.{rel['from_column']} "
                f"connects to {rel['to_table']}.{rel['to_column']}. "
                f"Business Logic: {rel['description']} "
                f"Usage: This relationship enables joining {rel['from_table']} and {rel['to_table']} "
                f"to analyze data across both entities."
            )
            documents.append(Document(
                text=relationship_text,
                metadata={
                    "source": "rules", 
                    "type": "relationship", 
                    "tables": [rel['from_table'], rel['to_table']],
                    "priority": "high"
                }
            ))

        # Enhanced formula processing with examples
        for formula in schema_data.get("golden_formulas", []):
            formula_text = f"Business Rule '{formula['name']}': {formula['description']}"
            
            if isinstance(formula.get('formula'), str):
                formula_text += f"\n\nSQL Implementation: {formula['formula']}"
            elif isinstance(formula.get('formula'), dict):
                formula_text += "\n\nDefinitions and Components:\n"
                formula_text += "\n".join(f"  • {k}: {v}" for k, v in formula['formula'].items())
            
            if formula.get('notes'):
                formula_text += f"\n\nImportant Notes: {formula['notes']}"
            
            # Add usage examples if available
            if formula.get('examples'):
                formula_text += f"\n\nUsage Examples: {formula['examples']}"

            documents.append(Document(
                text=formula_text,
                metadata={
                    "source": "rules", 
                    "type": "formula", 
                    "name": formula['name'],
                    "priority": "high"
                }
            ))

        if documents:
            # Use semantic splitter for better chunking
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95,
                embed_model=self.embedding_models["domain"]
            )
            
            index = VectorStoreIndex.from_documents(
                documents, 
                node_parser=node_parser,
                embed_model=self.embedding_models["domain"]
            )
            
            return index.as_retriever(similarity_top_k=3, embed_model=self.embedding_models["domain"])
        
        return None

    def _build_enhanced_schema_retriever(self, schema_data: Dict) -> BaseRetriever:
        """Build schema retriever with intelligent grouping."""
        inspector = inspect(self.engine)
        documents = []
        
        table_descriptions = schema_data.get("table_descriptions", {}) if schema_data else {}
        column_descriptions = schema_data.get("column_descriptions", {}) if schema_data else {}
        
        # Group related tables
        table_groups = self._group_related_tables(inspector.get_table_names())
        
        for group_name, tables in table_groups.items():
            group_doc_parts = [f"Table Group: {group_name}\n"]
            
            for table_name in tables:
                try:
                    columns = inspector.get_columns(table_name)
                    if not columns:
                        continue
                    
                    # Enhanced table description with statistics
                    table_desc = table_descriptions.get(table_name, f"Data table containing {self.db_stats.get(table_name, {}).get('row_count', 'unknown')} records")
                    
                    doc_part = f"\nTable '{table_name}': {table_desc}"
                    
                    # Add table statistics
                    if table_name in self.db_stats:
                        stats = self.db_stats[table_name]
                        doc_part += f" (Records: {stats['row_count']:,})"
                    
                    doc_part += "\nColumns:"
                    
                    # Enhanced column information
                    for col in columns:
                        col_name = col['name']
                        col_type = str(col['type'])
                        
                        # Get column description
                        col_desc = column_descriptions.get(table_name, {}).get(col_name, "")
                        
                        col_line = f"\n  • {col_name} ({col_type})"
                        if col_desc:
                            col_line += f": {col_desc}"
                        
                        # Add statistics if available
                        if table_name in self.db_stats and col_name in self.db_stats[table_name]['columns']:
                            col_stats = self.db_stats[table_name]['columns'][col_name]
                            if col_stats.get('distinct_count'):
                                col_line += f" [Distinct values: {col_stats['distinct_count']}]"
                            if col_stats.get('samples'):
                                col_line += f" [Examples: {', '.join(map(str, col_stats['samples'][:3]))}]"
                        
                        doc_part += col_line
                    
                    group_doc_parts.append(doc_part)
                    
                except Exception as e:
                    print(f"Warning: Could not process table '{table_name}': {e}")
            
            if len(group_doc_parts) > 1:  # More than just the group name
                documents.append(Document(
                    text="\n".join(group_doc_parts),
                    metadata={
                        "source": "schema", 
                        "group": group_name,
                        "tables": tables,
                        "priority": "high"
                    }
                ))
        
        if not documents:
            raise ValueError("No accessible database tables found")
        
        # Use sentence splitter for schema documents
        node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
        index = VectorStoreIndex.from_documents(documents, node_parser=node_parser)
        
        return index.as_retriever(similarity_top_k=5)

    def _group_related_tables(self, table_names: List[str]) -> Dict[str, List[str]]:
        """Intelligently group related tables."""
        groups = defaultdict(list)
        
        # Simple grouping based on name patterns
        for table_name in table_names:
            # Extract base name (remove common suffixes)
            base_name = table_name.lower()
            for suffix in ['_data', '_info', '_details', '_master', '_dim', '_fact']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Group by common prefixes
            prefix = base_name.split('_')[0] if '_' in base_name else base_name[:3]
            groups[prefix].append(table_name)
        
        # Ensure no group is too large
        final_groups = {}
        for group_name, tables in groups.items():
            if len(tables) <= 5:
                final_groups[group_name] = tables
            else:
                # Split large groups
                for i in range(0, len(tables), 5):
                    chunk_tables = tables[i:i+5]
                    final_groups[f"{group_name}_{i//5+1}"] = chunk_tables
        
        return final_groups

    def _build_smart_samples_retriever(self, schema_data: Dict) -> Optional[BaseRetriever]:
        """Build samples retriever with intelligent value selection."""
        documents = []
        column_descriptions = schema_data.get("column_descriptions", {}) if schema_data else {}
        
        inspector = inspect(self.engine)
        
        with self.engine.connect() as conn:
            for table_name in inspector.get_table_names():
                try:
                    columns = inspector.get_columns(table_name)
                    
                    for col in columns:
                        col_name = col["name"]
                        col_type_str = str(col["type"]).lower()
                        
                        # Focus on text/categorical columns with business value
                        if any(text_type in col_type_str for text_type in ["char", "text", "string", "varchar"]):
                            # Get diverse sample values using statistical sampling
                            if table_name in self.db_stats and col_name in self.db_stats[table_name]['columns']:
                                col_stats = self.db_stats[table_name]['columns'][col_name]
                                distinct_count = col_stats.get('distinct_count', 0)
                                
                                # Only process columns with reasonable cardinality
                                if 2 <= distinct_count <= 1000:
                                    # Use stratified sampling for better diversity
                                    sample_size = min(15, max(5, distinct_count // 10))
                                    
                                    # Get representative samples
                                    query = text(f'''
                                        SELECT "{col_name}", COUNT(*) as frequency
                                        FROM "{table_name}" 
                                        WHERE "{col_name}" IS NOT NULL 
                                        AND LENGTH(TRIM("{col_name}")) > 0
                                        GROUP BY "{col_name}"
                                        ORDER BY COUNT(*) DESC, "{col_name}"
                                        LIMIT :limit
                                    ''')
                                    
                                    result = conn.execute(query, {"limit": sample_size}).fetchall()
                                    values_with_freq = [(str(row[0]).strip(), row[1]) for row in result 
                                                       if row[0] is not None and str(row[0]).strip()]
                                    
                                    if values_with_freq:
                                        # Enhanced description with business context
                                        col_desc = column_descriptions.get(table_name, {}).get(col_name, "")
                                        desc_prefix = f"{col_desc}. " if col_desc else ""
                                        
                                        # Create comprehensive value documentation
                                        values_text = f"{desc_prefix}Column '{table_name}.{col_name}' contains categorical data. "
                                        values_text += f"Total distinct values: {distinct_count}. "
                                        
                                        # Group values by frequency for better context
                                        high_freq = [v for v, f in values_with_freq if f > col_stats.get('avg_freq', 1)]
                                        all_values = [v for v, f in values_with_freq]
                                        
                                        if high_freq:
                                            values_text += f"Most common values: {', '.join(high_freq[:5])}. "
                                        
                                        values_text += f"Sample values: {', '.join(all_values[:10])}"
                                        
                                        # Add business context hints
                                        if any(keyword in col_name.lower() for keyword in ['status', 'type', 'category', 'class']):
                                            values_text += f". This appears to be a classification field for filtering and grouping."
                                        
                                        documents.append(Document(
                                            text=values_text,
                                            metadata={
                                                "source": "values", 
                                                "table": table_name, 
                                                "column": col_name,
                                                "distinct_count": distinct_count,
                                                "priority": "medium" if distinct_count < 50 else "low"
                                            }
                                        ))
                                        
                except Exception as e:
                    print(f"Warning: Could not extract values from table '{table_name}': {e}")
        
        if documents:
            # Prioritize documents by business relevance
            documents.sort(key=lambda doc: (
                doc.metadata.get("priority", "low") == "high",
                doc.metadata.get("distinct_count", 0) < 100,
                doc.metadata.get("distinct_count", 0)
            ), reverse=True)
            
            # Limit to most relevant documents to manage index size
            documents = documents[:50]
            
            index = VectorStoreIndex.from_documents(documents)
            return index.as_retriever(similarity_top_k=5)
        
        return None

    def _build_statistics_retriever(self) -> Optional[BaseRetriever]:
        """Build retriever for database statistics and metadata."""
        documents = []
        
        for table_name, stats in self.db_stats.items():
            # Create comprehensive statistics document
            stats_text = f"Database Statistics for '{table_name}':\n"
            stats_text += f"Total Records: {stats['row_count']:,}\n"
            stats_text += f"Columns: {len(stats['columns'])}\n\n"
            
            # Numeric column statistics
            numeric_cols = []
            categorical_cols = []
            
            for col_name, col_stats in stats['columns'].items():
                if col_stats.get('min') is not None:  # Numeric column
                    numeric_cols.append(f"  • {col_name}: Range {col_stats['min']:,.0f} to {col_stats['max']:,.0f} "
                                      f"(Avg: {col_stats.get('avg', 0):,.1f}, Distinct: {col_stats.get('distinct_count', 0):,})")
                else:  # Categorical column
                    categorical_cols.append(f"  • {col_name}: {col_stats.get('distinct_count', 0):,} distinct values")
            
            if numeric_cols:
                stats_text += "Numeric Columns:\n" + "\n".join(numeric_cols) + "\n\n"
            
            if categorical_cols:
                stats_text += "Categorical Columns:\n" + "\n".join(categorical_cols) + "\n\n"
            
            # Add data quality insights
            total_cols = len(stats['columns'])
            numeric_ratio = len(numeric_cols) / total_cols if total_cols > 0 else 0
            
            if numeric_ratio > 0.5:
                stats_text += "Data Profile: This table contains primarily numeric data, suitable for aggregation and statistical analysis.\n"
            else:
                stats_text += "Data Profile: This table contains primarily categorical data, suitable for filtering and grouping operations.\n"
            
            documents.append(Document(
                text=stats_text,
                metadata={
                    "source": "statistics",
                    "table": table_name,
                    "row_count": stats['row_count'],
                    "priority": "medium"
                }
            ))
        
        if documents:
            index = VectorStoreIndex.from_documents(documents)
            return index.as_retriever(similarity_top_k=3)
        
        return None

    def _build_relationships_retriever(self) -> Optional[BaseRetriever]:
        """Build retriever for table relationships and join patterns."""
        relationships = self._detect_enhanced_relationships()
        
        if not relationships:
            return None
        
        documents = []
        
        for rel_key, rel_info in relationships.items():
            rel_text = f"Table Relationship: {rel_info['table1']} ↔ {rel_info['table2']}\n"
            rel_text += f"Join Condition: {rel_info['join_condition']}\n"
            rel_text += f"Relationship Type: {rel_info['relationship_type']}\n"
            
            # Add cardinality information if available
            if 'cardinality' in rel_info:
                rel_text += f"Cardinality: {rel_info['cardinality']}\n"
            
            # Add usage suggestions
            rel_text += f"Usage: Use this relationship to analyze data across {rel_info['table1']} and {rel_info['table2']} "
            rel_text += f"by joining on {rel_info['common_column']}.\n"
            
            # Add performance hints
            if rel_info.get('performance_hint'):
                rel_text += f"Performance Note: {rel_info['performance_hint']}\n"
            
            documents.append(Document(
                text=rel_text,
                metadata={
                    "source": "relationships",
                    "tables": [rel_info['table1'], rel_info['table2']],
                    "priority": "high"
                }
            ))
        
        if documents:
            index = VectorStoreIndex.from_documents(documents)
            return index.as_retriever(similarity_top_k=4)
        
        return None

    def _build_query_patterns_retriever(self) -> Optional[BaseRetriever]:
        """Build retriever for common query patterns and templates."""
        patterns = self._generate_query_patterns()
        
        documents = []
        
        for pattern in patterns:
            pattern_text = f"Query Pattern: {pattern['name']}\n"
            pattern_text += f"Description: {pattern['description']}\n"
            pattern_text += f"Use Case: {pattern['use_case']}\n"
            pattern_text += f"Template: {pattern['template']}\n"
            
            if 'examples' in pattern:
                pattern_text += f"Examples: {pattern['examples']}\n"
            
            documents.append(Document(
                text=pattern_text,
                metadata={
                    "source": "patterns",
                    "category": pattern['category'],
                    "priority": "medium"
                }
            ))
        
        if documents:
            index = VectorStoreIndex.from_documents(documents)
            return index.as_retriever(similarity_top_k=3)
        
        return None

    def _detect_enhanced_relationships(self) -> Dict[str, Dict]:
        """Enhanced relationship detection with cardinality analysis."""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            relationships = {}
            
            # Get all table columns with enhanced metadata
            table_columns = {}
            for table in tables:
                columns_info = {}
                for col in inspector.get_columns(table):
                    col_name = col['name']
                    columns_info[col_name] = {
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True)
                    }
                table_columns[table] = columns_info
            
            # Enhanced relationship detection
            with self.engine.connect() as conn:
                for table1 in tables:
                    for table2 in tables:
                        if table1 >= table2:  # Avoid duplicates and self-joins
                            continue
                        
                        # Find potential relationship columns
                        common_columns = set(table_columns[table1].keys()) & set(table_columns[table2].keys())
                        
                        for common_col in common_columns:
                            # Skip generic columns unless they're likely keys
                            if common_col.lower() in ['id', 'name', 'description', 'created_at', 'updated_at']:
                                if not any(suffix in common_col.upper() for suffix in ['KEY', 'ID', '_ID']):
                                    continue
                            
                            # Check if this looks like a foreign key relationship
                            if any(pattern in common_col.upper() for pattern in ['KEY', 'ID', 'NR', 'NUM', 'CODE']):
                                # Analyze cardinality
                                cardinality = self._analyze_cardinality(conn, table1, table2, common_col)
                                
                                if cardinality:  # Valid relationship found
                                    rel_key = f"{table1}_{table2}_{common_col}"
                                    relationships[rel_key] = {
                                        "table1": table1,
                                        "table2": table2, 
                                        "common_column": common_col,
                                        "join_condition": f"{table1}.{common_col} = {table2}.{common_col}",
                                        "relationship_type": cardinality['type'],
                                        "cardinality": cardinality['description'],
                                        "performance_hint": cardinality.get('hint', '')
                                    }
            
            return relationships
            
        except Exception as e:
            print(f"Warning: Enhanced relationship detection failed: {e}")
            return {}

    def _analyze_cardinality(self, conn, table1: str, table2: str, column: str) -> Optional[Dict]:
        """Analyze the cardinality of a potential relationship."""
        try:
            # Count distinct values in each table
            query1 = text(f'SELECT COUNT(DISTINCT "{column}") FROM "{table1}" WHERE "{column}" IS NOT NULL')
            query2 = text(f'SELECT COUNT(DISTINCT "{column}") FROM "{table2}" WHERE "{column}" IS NOT NULL')
            
            distinct1 = conn.execute(query1).scalar()
            distinct2 = conn.execute(query2).scalar()
            
            # Count total rows
            total1 = conn.execute(text(f'SELECT COUNT(*) FROM "{table1}"')).scalar()
            total2 = conn.execute(text(f'SELECT COUNT(*) FROM "{table2}"')).scalar()
            
            # Determine relationship type
            ratio1 = distinct1 / max(1, total1)  # Uniqueness ratio in table1
            ratio2 = distinct2 / max(1, total2)  # Uniqueness ratio in table2
            
            if ratio1 > 0.95 and ratio2 > 0.95:
                return {
                    'type': 'one_to_one',
                    'description': f'1:1 (Both tables have unique {column} values)',
                    'hint': 'Low join cost, consider which table to use as the primary'
                }
            elif ratio1 > 0.95:
                return {
                    'type': 'one_to_many',
                    'description': f'1:N ({table1} has unique {column}, {table2} may have duplicates)',
                    'hint': f'Join from {table1} to {table2} for detail expansion'
                }
            elif ratio2 > 0.95:
                return {
                    'type': 'many_to_one',
                    'description': f'N:1 ({table2} has unique {column}, {table1} may have duplicates)',
                    'hint': f'Join from {table2} to {table1} for aggregation'
                }
            else:
                # Check if there are actually matching values
                match_query = text(f'''
                    SELECT COUNT(*) FROM "{table1}" t1 
                    JOIN "{table2}" t2 ON t1."{column}" = t2."{column}"
                    WHERE t1."{column}" IS NOT NULL
                ''')
                matches = conn.execute(match_query).scalar()
                
                if matches > 0:
                    return {
                        'type': 'many_to_many',
                        'description': f'M:N (Both tables have non-unique {column} values, {matches} matching records)',
                        'hint': 'Consider if this join makes sense for your analysis'
                    }
            
            return None
            
        except Exception as e:
            print(f"Warning: Cardinality analysis failed for {table1}.{column} - {table2}.{column}: {e}")
            return None

    def _generate_query_patterns(self) -> List[Dict]:
        """Generate common query patterns based on database structure."""
        patterns = []
        
        # Basic patterns
        patterns.extend([
            {
                'name': 'Simple Count',
                'category': 'aggregation',
                'description': 'Count total records in a table',
                'use_case': 'Get total number of items, customers, orders, etc.',
                'template': 'SELECT COUNT(*) FROM {table} WHERE {conditions}',
                'examples': 'How many customers do we have? Count all orders.'
            },
            {
                'name': 'Group By Aggregation',
                'category': 'aggregation', 
                'description': 'Group data and calculate aggregates',
                'use_case': 'Analyze data by categories, time periods, or other groupings',
                'template': 'SELECT {group_by_column}, COUNT(*), SUM({metric_column}) FROM {table} GROUP BY {group_by_column}',
                'examples': 'Sales by region, Orders by month, Revenue by product category'
            },
            {
                'name': 'Top N Analysis',
                'category': 'ranking',
                'description': 'Find top performing items',
                'use_case': 'Identify best customers, top products, highest values',
                'template': 'SELECT {columns} FROM {table} ORDER BY {metric_column} DESC LIMIT {n}',
                'examples': 'Top 10 customers by revenue, Best selling products'
            },
            {
                'name': 'Time Series Analysis',
                'category': 'temporal',
                'description': 'Analyze data over time periods',
                'use_case': 'Track trends, growth, seasonal patterns',
                'template': 'SELECT DATE_TRUNC(\'{period}\', {date_column}), COUNT(*) FROM {table} GROUP BY DATE_TRUNC(\'{period}\', {date_column}) ORDER BY 1',
                'examples': 'Monthly sales trends, Weekly user activity'
            }
        ])
        
        # Add database-specific patterns based on detected structure
        if self.db_stats:
            # Look for common table patterns
            table_names = list(self.db_stats.keys())
            
            # Financial patterns
            financial_tables = [t for t in table_names if any(word in t.lower() for word in ['sales', 'revenue', 'invoice', 'payment', 'transaction'])]
            if financial_tables:
                patterns.append({
                    'name': 'Revenue Analysis',
                    'category': 'financial',
                    'description': 'Calculate total revenue and related metrics',
                    'use_case': 'Financial reporting and analysis',
                    'template': f'SELECT SUM(amount), COUNT(*) FROM {financial_tables[0]} WHERE date >= \'{{start_date}}\'',
                    'examples': 'Total revenue this quarter, Average transaction value'
                })
            
            # Inventory patterns  
            inventory_tables = [t for t in table_names if any(word in t.lower() for word in ['inventory', 'stock', 'product', 'item'])]
            if inventory_tables:
                patterns.append({
                    'name': 'Inventory Status',
                    'category': 'inventory',
                    'description': 'Check stock levels and availability',
                    'use_case': 'Inventory management and restocking',
                    'template': f'SELECT product_name, stock_level FROM {inventory_tables[0]} WHERE stock_level < {{threshold}}',
                    'examples': 'Low stock items, Out of stock products'
                })
        
        return patterns

    def intelligent_retrieve(self, context: RetrievalContext) -> str:
        """Main retrieval method with intelligent layer selection."""
        
        # Create cache key for this query context
        cache_key = self._create_context_cache_key(context)
        
        # Check cache first
        if cache_key in self.context_cache and self.rag_config.get('use_cache', True):
            return self.context_cache[cache_key]
        
        # Select appropriate retrieval layers based on query classification
        active_layers = self._select_retrieval_layers(context)
        
        # Retrieve from each active layer
        context_parts = []
        remaining_budget = context.token_budget
        
        for layer, priority in active_layers:
            if remaining_budget <= 0:
                break
                
            if layer in self.retrievers and self.retrievers[layer]:
                try:
                    # Adjust retrieval parameters based on priority and budget
                    top_k = min(5, max(1, remaining_budget // 200))  # Estimate 200 tokens per result
                    
                    nodes = self.retrievers[layer].retrieve(context.query)[:top_k]
                    
                    if nodes:
                        layer_content = self._process_retrieval_results(nodes, layer, remaining_budget)
                        if layer_content:
                            context_parts.append({
                                'layer': layer.value,
                                'content': layer_content,
                                'priority': priority,
                                'token_count': len(layer_content.split()) * 1.3
                            })
                            remaining_budget -= len(layer_content.split()) * 1.3
                
                except Exception as e:
                    print(f"Warning: Retrieval failed for layer {layer.value}: {e}")
        
        # Combine and optimize context
        final_context = self._combine_context_parts(context_parts, context)
        
        # Cache the result
        if self.rag_config.get('use_cache', True):
            self.context_cache[cache_key] = final_context
        
        return final_context

    def _select_retrieval_layers(self, context: RetrievalContext) -> List[Tuple[RetrievalLayer, str]]:
        """Select appropriate retrieval layers based on query classification."""
        classification = context.classification
        layers = []
        
        # Always include schema (highest priority)
        layers.append((RetrievalLayer.SCHEMA, "high"))
        
        # Business rules for complex queries or when business context is needed
        if (classification.complexity.value in ['moderate', 'complex'] or 
            classification.intent.value in ['analysis', 'relationship'] or
            any(word in context.query.lower() for word in ['calculate', 'formula', 'rule', 'policy'])):
            layers.append((RetrievalLayer.BUSINESS_RULES, "high"))
        
        # Samples for lookup queries or when specific values are mentioned
        if (classification.intent.value == 'lookup' or
            any(word in context.query.lower() for word in ['like', 'contains', 'named', 'called'])):
            layers.append((RetrievalLayer.SAMPLES, "medium"))
        
        # Statistics for analytical queries
        if classification.intent.value in ['analysis', 'aggregation', 'trend']:
            layers.append((RetrievalLayer.STATISTICS, "medium"))
        
        # Relationships for multi-table queries
        if (classification.complexity.value in ['moderate', 'complex'] or
            'join' in context.query.lower() or
            len(context.table_context) > 1):
            layers.append((RetrievalLayer.RELATIONSHIPS, "high"))
        
        # Query patterns for complex analytical requests
        if classification.complexity.value == 'complex':
            layers.append((RetrievalLayer.QUERY_PATTERNS, "low"))
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        layers.sort(key=lambda x: priority_order[x[1]], reverse=True)
        
        return layers

    def _process_retrieval_results(self, nodes, layer: RetrievalLayer, budget: int) -> str:
        """Process and optimize retrieval results for a specific layer."""
        if not nodes:
            return ""
        
        contents = []
        current_budget = budget
        
        for node in nodes:
            content = node.get_content()
            content_tokens = len(content.split()) * 1.3
            
            if content_tokens > current_budget:
                # Truncate if necessary
                words = content.split()
                max_words = int(current_budget / 1.3)
                content = ' '.join(words[:max_words]) + "..."
                content_tokens = max_words * 1.3
            
            contents.append(content)
            current_budget -= content_tokens
            
            if current_budget <= 0:
                break
        
        return '\n'.join(contents)

    def _combine_context_parts(self, context_parts: List[Dict], context: RetrievalContext) -> str:
        """Combine context parts into final context string."""
        if not context_parts:
            return "No relevant context found."
        
        # Sort by priority and relevance
        context_parts.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}[x['priority']],
            -x['token_count']  # Prefer more substantial content
        ), reverse=True)
        
        # Build final context with proper sectioning
        final_parts = []
        
        layer_headers = {
            'business_rules': '### Business Rules & Context',
            'schema': '### Database Schema',
            'samples': '### Sample Data Values', 
            'statistics': '### Database Statistics',
            'relationships': '### Table Relationships',
            'query_patterns': '### Query Patterns'
        }
        
        for part in context_parts:
            header = layer_headers.get(part['layer'], f"### {part['layer'].title()}")
            final_parts.append(f"{header}\n{part['content']}")
        
        result = "\n\n---\n\n".join(final_parts)
        
        # Final token budget check
        estimated_tokens = len(result.split()) * 1.3
        if estimated_tokens > context.token_budget:
            # Proportionally reduce each section
            reduction_factor = context.token_budget / estimated_tokens
            truncated_parts = []
            
            for part in final_parts:
                target_length = int(len(part) * reduction_factor)
                if len(part) > target_length:
                    truncated_parts.append(part[:target_length] + "...")
                else:
                    truncated_parts.append(part)
            
            result = "\n\n---\n\n".join(truncated_parts)
            result += "\n\n[Context optimized for token budget]"
        
        return result

    def _create_context_cache_key(self, context: RetrievalContext) -> str:
        """Create a cache key for context retrieval."""
        key_components = [
            context.query.lower().strip(),
            context.classification.complexity.value,
            context.classification.intent.value,
            str(sorted(context.table_context.keys())) if context.table_context else "",
            str(context.token_budget)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_key(self, schema_data: Optional[Dict]) -> str:
        """Generate cache key for retriever storage."""
        if not schema_data:
            return "basic"
        
        key_components = [
            str(sorted(schema_data.get("table_descriptions", {}).keys())),
            str(len(schema_data.get("relationships", []))),
            str(len(schema_data.get("golden_formulas", [])))
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:8]

    def _cache_retrievers(self, cache_path: Path):
        """Cache retrievers for future use."""
        # This is a placeholder - in practice, you'd implement proper serialization
        # of the vector indices using llama-index's built-in persistence
        pass

    def _load_cached_retrievers(self, cache_path: Path) -> Dict[RetrievalLayer, Optional[BaseRetriever]]:
        """Load cached retrievers."""
        # This is a placeholder - implement proper deserialization
        return self.retrievers

# Convenience function for backward compatibility
def build_scalable_retriever_system(engine: Engine, schema_data: Optional[Dict] = None):
    """Build the enhanced RAG system and return retrievers in legacy format."""
    rag_system = HierarchicalRAGSystem(engine)
    retrievers_dict = rag_system.build_enhanced_retrievers(schema_data)
    
    # Convert to legacy format
    return (
        retrievers_dict.get(RetrievalLayer.BUSINESS_RULES),
        retrievers_dict.get(RetrievalLayer.SCHEMA), 
        retrievers_dict.get(RetrievalLayer.SAMPLES)
    )

def combine_retriever_results(retrievers, query: str, max_context_length: int = 4000) -> str:
    """Enhanced context combination with intelligent retrieval."""
    from .query_classifier import get_query_classifier
    
    # Classify the query for intelligent retrieval
    classifier = get_query_classifier()
    classification = classifier.classify_query(query)
    
    # Create retrieval context
    context = RetrievalContext(
        query=query,
        classification=classification,
        table_context={},  # Would be populated with actual table context
        token_budget=max_context_length
    )
    
    # If we have the old format retrievers, convert them
    if isinstance(retrievers, tuple) and len(retrievers) == 3:
        rules_retriever, schema_retriever, values_retriever = retrievers
        
        # Create a simple RAG system for backward compatibility
        rag_system = HierarchicalRAGSystem(None)  # Engine not needed for this path
        rag_system.retrievers = {
            RetrievalLayer.BUSINESS_RULES: rules_retriever,
            RetrievalLayer.SCHEMA: schema_retriever,
            RetrievalLayer.SAMPLES: values_retriever
        }
        
        return rag_system.intelligent_retrieve(context)
    
    return "No relevant context found."