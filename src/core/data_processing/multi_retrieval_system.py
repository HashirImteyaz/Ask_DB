# src/core/data_processing/multi_retrieval_system.py

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import defaultdict

from sqlalchemy import Engine, inspect, text

# Optional imports with fallbacks
try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
    from llama_index.embeddings.openai import OpenAIEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    print("LlamaIndex not available - multi-retrieval system will use fallback")
    BaseRetriever = object
    Document = object
    VectorStoreIndex = object
    LLAMA_INDEX_AVAILABLE = False


class RetrievalType(Enum):
    """Different types of retrieval for different content types."""
    TABLE_DESCRIPTIONS = "table_descriptions"
    COLUMN_DESCRIPTIONS = "column_descriptions"
    HYBRID = "hybrid"


@dataclass
class MultiRetrievalContext:
    """Context for multi-retrieval operations."""
    query: str
    retrieval_types: List[RetrievalType]
    table_context: Dict
    token_budget: int
    classification: Any  # QueryClassification
    focus_tables: Optional[List[str]] = None
    focus_columns: Optional[List[str]] = None


@dataclass
class RetrievalResult:
    """Result from a specific retrieval system."""
    retrieval_type: RetrievalType
    content: str
    confidence: float
    metadata: Dict
    token_count: int


class ColumnDescriptionRetriever:
    """Specialized retriever for column descriptions and metadata."""
    
    def __init__(self, engine: Engine, schema_data: Optional[Dict] = None):
        self.engine = engine
        self.schema_data = schema_data or {}
        self.column_index = None
        self.embedding_model = None
        
        if LLAMA_INDEX_AVAILABLE:
            self.embedding_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                dimensions=1536
            )
            self._build_column_index()
    
    def _build_column_index(self):
        """Build specialized vector index for column descriptions."""
        documents = []
        column_descriptions = self.schema_data.get("column_descriptions", {})
        
        # Print extracted column descriptions from schema
        print("\n🔍 EXTRACTING COLUMN DESCRIPTIONS FROM SCHEMA:")
        print("=" * 60)
        if column_descriptions:
            for table_name, columns in column_descriptions.items():
                print(f"📋 Table: {table_name}")
                for col_name, col_desc in columns.items():
                    print(f"  └── Column: {col_name} → {col_desc}")
                print()
        else:
            print("⚠️ No column descriptions found in schema data")
        print("=" * 60)
        
        inspector = inspect(self.engine)
        
        with self.engine.connect() as conn:
            # Print database structure for comparison
            db_tables = inspector.get_table_names()
            print(f"\n🗃️ DATABASE STRUCTURE DETECTED:")
            print("=" * 50)
            print(f"📋 Tables in database: {len(db_tables)}")
            for table_name in db_tables:
                columns = inspector.get_columns(table_name)
                print(f"  └── {table_name} ({len(columns)} columns)")
                for col in columns[:3]:  # Show first 3 columns
                    print(f"      • {col['name']} ({col['type']})")
                if len(columns) > 3:
                    print(f"      ... and {len(columns) - 3} more columns")
            print("=" * 50)
            
            print(f"\n🔄 PROCESSING COLUMN DESCRIPTIONS:")
            print("=" * 50)
            for table_name in inspector.get_table_names():
                try:
                    columns = inspector.get_columns(table_name)
                    
                    for col in columns:
                        col_name = col['name']
                        col_type = str(col['type'])
                        
                        # Get column description from schema data
                        col_desc = column_descriptions.get(table_name, {}).get(col_name, "")
                        
                        # Print extraction details for each column
                        if col_desc:
                            print(f"✅ MATCHED: {table_name}.{col_name} → '{col_desc}'")
                        else:
                            print(f"❌ NO DESC: {table_name}.{col_name} (using DB introspection only)")
                        
                        # Create comprehensive column document
                        col_text = f"Column: {table_name}.{col_name}\n"
                        col_text += f"Data Type: {col_type}\n"
                        
                        if col_desc:
                            col_text += f"Description: {col_desc}\n"
                        
                        # Add data statistics
                        try:
                            # Get basic statistics
                            if 'int' in col_type.lower() or 'float' in col_type.lower() or 'decimal' in col_type.lower():
                                stats_query = text(f"""
                                    SELECT 
                                        MIN("{col_name}") as min_val,
                                        MAX("{col_name}") as max_val,
                                        AVG("{col_name}") as avg_val,
                                        COUNT(DISTINCT "{col_name}") as distinct_count
                                    FROM "{table_name}"
                                    WHERE "{col_name}" IS NOT NULL
                                """)
                                stats = conn.execute(stats_query).fetchone()
                                if stats:
                                    col_text += f"Range: {stats[0]} to {stats[1]}\n"
                                    col_text += f"Average: {stats[2]:.2f}\n"
                                    col_text += f"Distinct Values: {stats[3]}\n"
                            else:
                                # For categorical/text columns
                                sample_query = text(f"""
                                    SELECT DISTINCT "{col_name}" 
                                    FROM "{table_name}" 
                                    WHERE "{col_name}" IS NOT NULL 
                                    LIMIT 5
                                """)
                                samples = [row[0] for row in conn.execute(sample_query).fetchall()]
                                if samples:
                                    col_text += f"Sample Values: {', '.join(map(str, samples))}\n"
                                
                                # Get distinct count
                                count_query = text(f"""
                                    SELECT COUNT(DISTINCT "{col_name}") 
                                    FROM "{table_name}" 
                                    WHERE "{col_name}" IS NOT NULL
                                """)
                                distinct_count = conn.execute(count_query).scalar()
                                col_text += f"Distinct Values: {distinct_count}\n"
                                
                        except Exception as e:
                            print(f"Warning: Could not get statistics for {table_name}.{col_name}: {e}")
                        
                        # Add business context hints
                        business_hints = self._generate_business_hints(col_name, col_type, col_desc)
                        if business_hints:
                            col_text += f"Business Context: {business_hints}\n"
                        
                        documents.append(Document(
                            text=col_text,
                            metadata={
                                "type": "column",
                                "table": table_name,
                                "column": col_name,
                                "data_type": col_type,
                                "has_description": bool(col_desc),
                                "priority": self._calculate_column_priority(col_name, col_type, col_desc)
                            }
                        ))
                        
                except Exception as e:
                    print(f"Warning: Could not process columns for table '{table_name}': {e}")
        
        if documents:
            # Use semantic splitter for better column understanding
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embedding_model
            )
            
            self.column_index = VectorStoreIndex.from_documents(
                documents,
                node_parser=node_parser,
                embed_model=self.embedding_model
            )
    
    def _generate_business_hints(self, col_name: str, col_type: str, col_desc: str) -> str:
        """Generate business context hints for columns."""
        hints = []
        col_lower = col_name.lower()
        
        # ID columns
        if 'id' in col_lower or col_lower.endswith('_id'):
            hints.append("Primary identifier or foreign key")
        
        # Date/time columns
        if any(word in col_lower for word in ['date', 'time', 'created', 'updated', 'modified']):
            hints.append("Temporal field for time-based analysis")
        
        # Status/category columns
        if any(word in col_lower for word in ['status', 'type', 'category', 'class', 'state']):
            hints.append("Categorical field for filtering and grouping")
        
        # Amount/quantity columns
        if any(word in col_lower for word in ['amount', 'price', 'cost', 'total', 'quantity', 'count']):
            hints.append("Numeric measure for aggregation")
        
        # Name/description columns
        if any(word in col_lower for word in ['name', 'title', 'description', 'label']):
            hints.append("Descriptive text field")
        
        return "; ".join(hints)
    
    def _calculate_column_priority(self, col_name: str, col_type: str, col_desc: str) -> str:
        """Calculate priority level for column retrieval."""
        col_lower = col_name.lower()
        
        # High priority for key business columns
        if any(word in col_lower for word in ['id', 'name', 'title', 'status', 'date', 'amount', 'price']):
            return "high"
        
        # High priority if has description
        if col_desc:
            return "high"
        
        # Medium priority for common business fields
        if any(word in col_lower for word in ['type', 'category', 'count', 'total', 'description']):
            return "medium"
        
        return "low"
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant column information."""
        if not self.column_index:
            return []
        
        try:
            retriever = self.column_index.as_retriever(
                similarity_top_k=top_k,
                embed_model=self.embedding_model
            )
            nodes = retriever.retrieve(query)
            
            results = []
            for node in nodes:
                results.append({
                    'content': node.text,
                    'metadata': node.metadata,
                    'score': getattr(node, 'score', 0.0)
                })
            
            return results
            
        except Exception as e:
            print(f"Column retrieval error: {e}")
            return []


class TableDescriptionRetriever:
    """Specialized retriever for table descriptions and high-level schema information."""
    
    def __init__(self, engine: Engine, schema_data: Optional[Dict] = None):
        self.engine = engine
        self.schema_data = schema_data or {}
        self.table_index = None
        self.embedding_model = None
        
        if LLAMA_INDEX_AVAILABLE:
            self.embedding_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                dimensions=1536
            )
            self._build_table_index()
    
    def _build_table_index(self):
        """Build specialized vector index for table descriptions."""
        documents = []
        table_descriptions = self.schema_data.get("table_descriptions", {})
        
        # Print extracted table descriptions from schema
        print("\n🏗️ EXTRACTING TABLE DESCRIPTIONS FROM SCHEMA:")
        print("=" * 60)
        if table_descriptions:
            for table_name, table_desc in table_descriptions.items():
                print(f"📋 Table: {table_name} → '{table_desc}'")
        else:
            print("⚠️ No table descriptions found in schema data")
        print("=" * 60)
        
        inspector = inspect(self.engine)
        
        with self.engine.connect() as conn:
            print(f"\n🔄 PROCESSING TABLE DESCRIPTIONS:")
            print("=" * 50)
            for table_name in inspector.get_table_names():
                try:
                    # Get table description from schema data
                    table_desc = table_descriptions.get(table_name, "")
                    
                    # Print extraction details for each table
                    if table_desc:
                        print(f"✅ MATCHED: Table '{table_name}' → '{table_desc}'")
                    else:
                        print(f"❌ NO DESC: Table '{table_name}' (using DB introspection only)")
                    
                    # Create comprehensive table document
                    table_text = f"Table: {table_name}\n"
                    
                    if table_desc:
                        table_text += f"Description: {table_desc}\n"
                    
                    # Add table statistics
                    try:
                        count_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                        row_count = conn.execute(count_query).scalar()
                        table_text += f"Total Records: {row_count:,}\n"
                    except Exception as e:
                        print(f"Warning: Could not get row count for {table_name}: {e}")
                    
                    # Add column summary
                    columns = inspector.get_columns(table_name)
                    table_text += f"Number of Columns: {len(columns)}\n"
                    
                    # Categorize columns
                    id_columns = [col['name'] for col in columns if 'id' in col['name'].lower()]
                    date_columns = [col['name'] for col in columns if any(word in col['name'].lower() for word in ['date', 'time', 'created', 'updated'])]
                    amount_columns = [col['name'] for col in columns if any(word in col['name'].lower() for word in ['amount', 'price', 'cost', 'total', 'quantity'])]
                    
                    if id_columns:
                        table_text += f"ID Columns: {', '.join(id_columns)}\n"
                    if date_columns:
                        table_text += f"Date Columns: {', '.join(date_columns)}\n"
                    if amount_columns:
                        table_text += f"Amount Columns: {', '.join(amount_columns)}\n"
                    
                    # Add business context
                    business_context = self._generate_table_business_context(table_name, columns, table_desc)
                    if business_context:
                        table_text += f"Business Context: {business_context}\n"
                    
                    # Add relationship hints
                    relationship_hints = self._generate_relationship_hints(table_name, columns)
                    if relationship_hints:
                        table_text += f"Relationships: {relationship_hints}\n"
                    
                    documents.append(Document(
                        text=table_text,
                        metadata={
                            "type": "table",
                            "table": table_name,
                            "row_count": row_count if 'row_count' in locals() else 0,
                            "column_count": len(columns),
                            "has_description": bool(table_desc),
                            "priority": self._calculate_table_priority(table_name, table_desc, row_count if 'row_count' in locals() else 0)
                        }
                    ))
                    
                except Exception as e:
                    print(f"Warning: Could not process table '{table_name}': {e}")
        
        if documents:
            # Use sentence splitter for table documents
            node_parser = SentenceSplitter(
                chunk_size=800,
                chunk_overlap=50
            )
            
            self.table_index = VectorStoreIndex.from_documents(
                documents,
                node_parser=node_parser,
                embed_model=self.embedding_model
            )
    
    def _generate_table_business_context(self, table_name: str, columns: List[Dict], table_desc: str) -> str:
        """Generate business context for tables."""
        context = []
        table_lower = table_name.lower()
        
        # Identify table purpose based on name and columns
        if any(word in table_lower for word in ['order', 'sale', 'transaction', 'purchase']):
            context.append("Transactional data for sales/order analysis")
        elif any(word in table_lower for word in ['customer', 'client', 'user']):
            context.append("Customer/user master data")
        elif any(word in table_lower for word in ['product', 'item', 'inventory']):
            context.append("Product/inventory master data")
        elif any(word in table_lower for word in ['employee', 'staff', 'person']):
            context.append("Employee/personnel data")
        
        # Identify data pattern
        has_id = any('id' in col['name'].lower() for col in columns)
        has_dates = any(word in col['name'].lower() for col in columns for word in ['date', 'time', 'created'])
        has_amounts = any(word in col['name'].lower() for col in columns for word in ['amount', 'price', 'cost', 'total'])
        
        if has_id and has_dates and has_amounts:
            context.append("Contains transactional records with identifiers, timestamps, and monetary values")
        elif has_id and has_dates:
            context.append("Event or activity tracking table")
        elif has_id:
            context.append("Reference or master data table")
        
        return "; ".join(context)
    
    def _generate_relationship_hints(self, table_name: str, columns: List[Dict]) -> str:
        """Generate relationship hints for tables."""
        hints = []
        
        # Look for foreign key patterns
        fk_columns = [col['name'] for col in columns if col['name'].lower().endswith('_id') and col['name'].lower() != table_name.lower() + '_id']
        
        if fk_columns:
            hints.append(f"Links to other tables via: {', '.join(fk_columns)}")
        
        # Look for common join columns
        common_joins = [col['name'] for col in columns if col['name'].lower() in ['customer_id', 'product_id', 'order_id', 'user_id']]
        if common_joins:
            hints.append(f"Common join points: {', '.join(common_joins)}")
        
        return "; ".join(hints)
    
    def _calculate_table_priority(self, table_name: str, table_desc: str, row_count: int) -> str:
        """Calculate priority level for table retrieval."""
        table_lower = table_name.lower()
        
        # High priority for key business tables
        if any(word in table_lower for word in ['order', 'sale', 'customer', 'product', 'transaction']):
            return "high"
        
        # High priority if has description
        if table_desc:
            return "high"
        
        # High priority for large tables (likely important business data)
        if row_count > 1000:
            return "high"
        
        # Medium priority for moderate size tables
        if row_count > 100:
            return "medium"
        
        return "low"
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant table information."""
        if not self.table_index:
            return []
        
        try:
            retriever = self.table_index.as_retriever(
                similarity_top_k=top_k,
                embed_model=self.embedding_model
            )
            nodes = retriever.retrieve(query)
            
            results = []
            for node in nodes:
                results.append({
                    'content': node.text,
                    'metadata': node.metadata,
                    'score': getattr(node, 'score', 0.0)
                })
            
            return results
            
        except Exception as e:
            print(f"Table retrieval error: {e}")
            return []


class MultiRetrievalSystem:
    """Orchestrates multiple specialized retrievers for different content types."""
    
    def __init__(self, engine: Engine, schema_data: Optional[Dict] = None, config: Optional[Dict] = None):
        self.engine = engine
        self.schema_data = schema_data or {}
        self.config = config or {}
        
        # Print schema data summary
        self._print_schema_summary()
        
        # Initialize specialized retrievers
        self.column_retriever = ColumnDescriptionRetriever(engine, schema_data)
        self.table_retriever = TableDescriptionRetriever(engine, schema_data)
        
        # Configuration
        self.default_token_budget = self.config.get('default_token_budget', 3000)
        self.column_weight = self.config.get('column_weight', 0.6)
        self.table_weight = self.config.get('table_weight', 0.4)
        
        # Print initialization summary
        self._print_initialization_summary()
    
    def _print_schema_summary(self):
        """Print summary of schema data provided."""
        print("\n🚀 INITIALIZING MULTIPLE RETRIEVAL SYSTEM")
        print("=" * 70)
        
        table_descriptions = self.schema_data.get("table_descriptions", {})
        column_descriptions = self.schema_data.get("column_descriptions", {})
        
        print(f"📊 Schema Data Summary:")
        print(f"  • Table descriptions: {len(table_descriptions)} tables")
        print(f"  • Column descriptions: {sum(len(cols) for cols in column_descriptions.values())} columns across {len(column_descriptions)} tables")
        
        if table_descriptions:
            print(f"  • Tables with descriptions: {', '.join(table_descriptions.keys())}")
        
        if column_descriptions:
            print(f"  • Tables with column descriptions: {', '.join(column_descriptions.keys())}")
        
        print("=" * 70)
    
    def _print_initialization_summary(self):
        """Print summary after initialization."""
        print("\n✅ MULTIPLE RETRIEVAL SYSTEM INITIALIZED")
        print("=" * 70)
        print(f"🔧 Configuration:")
        print(f"  • Default token budget: {self.default_token_budget}")
        print(f"  • Column retriever weight: {self.column_weight}")
        print(f"  • Table retriever weight: {self.table_weight}")
        print(f"🎯 Specialized retrievers ready:")
        print(f"  • ColumnDescriptionRetriever: {'✅ Active' if self.column_retriever.column_index else '❌ No index'}")
        print(f"  • TableDescriptionRetriever: {'✅ Active' if self.table_retriever.table_index else '❌ No index'}")
        print("=" * 70)
    
    def retrieve(self, context: MultiRetrievalContext) -> Dict[RetrievalType, RetrievalResult]:
        """Main retrieval method using multiple specialized retrievers."""
        results = {}
        
        # Calculate token allocation
        column_budget = int(context.token_budget * self.column_weight)
        table_budget = int(context.token_budget * self.table_weight)
        
        # Retrieve from column descriptions
        if RetrievalType.COLUMN_DESCRIPTIONS in context.retrieval_types:
            column_result = self._retrieve_column_descriptions(
                context.query, 
                column_budget,
                context.focus_columns
            )
            if column_result:
                results[RetrievalType.COLUMN_DESCRIPTIONS] = column_result
        
        # Retrieve from table descriptions
        if RetrievalType.TABLE_DESCRIPTIONS in context.retrieval_types:
            table_result = self._retrieve_table_descriptions(
                context.query,
                table_budget,
                context.focus_tables
            )
            if table_result:
                results[RetrievalType.TABLE_DESCRIPTIONS] = table_result
        
        # Hybrid retrieval if requested
        if RetrievalType.HYBRID in context.retrieval_types:
            hybrid_result = self._retrieve_hybrid(context)
            if hybrid_result:
                results[RetrievalType.HYBRID] = hybrid_result
        
        return results
    
    def _retrieve_column_descriptions(self, query: str, token_budget: int, focus_columns: Optional[List[str]] = None) -> Optional[RetrievalResult]:
        """Retrieve column-specific information."""
        # Determine number of results based on token budget
        max_results = min(8, max(3, token_budget // 200))
        
        # Enhance query for column retrieval
        column_query = self._enhance_query_for_columns(query)
        
        results = self.column_retriever.retrieve(column_query, top_k=max_results)
        
        if not results:
            return None
        
        # Filter results if focus columns specified
        if focus_columns:
            filtered_results = []
            for result in results:
                if any(col in result['metadata'].get('column', '') for col in focus_columns):
                    filtered_results.append(result)
            results = filtered_results or results  # Fallback to all results if no matches
        
        # Combine and format results
        content_parts = []
        total_tokens = 0
        
        for result in results:
            content = result['content']
            estimated_tokens = len(content.split()) * 1.3
            
            if total_tokens + estimated_tokens <= token_budget:
                content_parts.append(content)
                total_tokens += estimated_tokens
            else:
                break
        
        combined_content = "\n\n".join(content_parts)
        
        return RetrievalResult(
            retrieval_type=RetrievalType.COLUMN_DESCRIPTIONS,
            content=combined_content,
            confidence=self._calculate_confidence(results),
            metadata={
                "source": "column_retriever",
                "num_results": len(content_parts),
                "focus_applied": bool(focus_columns)
            },
            token_count=int(total_tokens)
        )
    
    def _retrieve_table_descriptions(self, query: str, token_budget: int, focus_tables: Optional[List[str]] = None) -> Optional[RetrievalResult]:
        """Retrieve table-specific information."""
        # Determine number of results based on token budget
        max_results = min(5, max(2, token_budget // 300))
        
        # Enhance query for table retrieval
        table_query = self._enhance_query_for_tables(query)
        
        results = self.table_retriever.retrieve(table_query, top_k=max_results)
        
        if not results:
            return None
        
        # Filter results if focus tables specified
        if focus_tables:
            filtered_results = []
            for result in results:
                if any(table in result['metadata'].get('table', '') for table in focus_tables):
                    filtered_results.append(result)
            results = filtered_results or results  # Fallback to all results if no matches
        
        # Combine and format results
        content_parts = []
        total_tokens = 0
        
        for result in results:
            content = result['content']
            estimated_tokens = len(content.split()) * 1.3
            
            if total_tokens + estimated_tokens <= token_budget:
                content_parts.append(content)
                total_tokens += estimated_tokens
            else:
                break
        
        combined_content = "\n\n".join(content_parts)
        
        return RetrievalResult(
            retrieval_type=RetrievalType.TABLE_DESCRIPTIONS,
            content=combined_content,
            confidence=self._calculate_confidence(results),
            metadata={
                "source": "table_retriever",
                "num_results": len(content_parts),
                "focus_applied": bool(focus_tables)
            },
            token_count=int(total_tokens)
        )
    
    def _retrieve_hybrid(self, context: MultiRetrievalContext) -> Optional[RetrievalResult]:
        """Perform hybrid retrieval combining both table and column information."""
        # Split budget between table and column retrieval
        table_budget = context.token_budget // 3
        column_budget = (context.token_budget * 2) // 3
        
        # Get results from both retrievers
        table_results = self.table_retriever.retrieve(context.query, top_k=3)
        column_results = self.column_retriever.retrieve(context.query, top_k=6)
        
        if not table_results and not column_results:
            return None
        
        # Interleave results based on relevance and priority
        content_parts = []
        total_tokens = 0
        
        # Add table context first (higher level overview)
        table_content = []
        for result in table_results[:2]:  # Limit to top 2 tables
            content = result['content']
            estimated_tokens = len(content.split()) * 1.3
            if total_tokens + estimated_tokens <= table_budget:
                table_content.append(content)
                total_tokens += estimated_tokens
        
        if table_content:
            content_parts.append("=== TABLE INFORMATION ===\n" + "\n\n".join(table_content))
        
        # Add column details
        column_content = []
        remaining_budget = context.token_budget - total_tokens
        for result in column_results:
            content = result['content']
            estimated_tokens = len(content.split()) * 1.3
            if total_tokens + estimated_tokens <= remaining_budget:
                column_content.append(content)
                total_tokens += estimated_tokens
        
        if column_content:
            content_parts.append("=== COLUMN INFORMATION ===\n" + "\n\n".join(column_content))
        
        combined_content = "\n\n".join(content_parts)
        
        return RetrievalResult(
            retrieval_type=RetrievalType.HYBRID,
            content=combined_content,
            confidence=(self._calculate_confidence(table_results) + self._calculate_confidence(column_results)) / 2,
            metadata={
                "source": "hybrid_retriever",
                "table_results": len(table_content),
                "column_results": len(column_content)
            },
            token_count=int(total_tokens)
        )
    
    def _enhance_query_for_columns(self, query: str) -> str:
        """Enhance query specifically for column retrieval."""
        enhancements = []
        query_lower = query.lower()
        
        # Add column-specific keywords
        if any(word in query_lower for word in ['field', 'attribute', 'property', 'data type', 'column']):
            enhancements.append("column information field details")
        
        if any(word in query_lower for word in ['values', 'examples', 'samples', 'range']):
            enhancements.append("data values examples range statistics")
        
        if any(word in query_lower for word in ['description', 'meaning', 'purpose']):
            enhancements.append("column description meaning purpose")
        
        enhanced_query = query
        if enhancements:
            enhanced_query += " " + " ".join(enhancements)
        
        return enhanced_query
    
    def _enhance_query_for_tables(self, query: str) -> str:
        """Enhance query specifically for table retrieval."""
        enhancements = []
        query_lower = query.lower()
        
        # Add table-specific keywords
        if any(word in query_lower for word in ['table', 'entity', 'schema', 'structure']):
            enhancements.append("table information schema structure")
        
        if any(word in query_lower for word in ['relationship', 'join', 'connect', 'link']):
            enhancements.append("table relationships joins connections")
        
        if any(word in query_lower for word in ['purpose', 'business', 'function']):
            enhancements.append("table purpose business function")
        
        enhanced_query = query
        if enhancements:
            enhanced_query += " " + " ".join(enhancements)
        
        return enhanced_query
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not results:
            return 0.0
        
        # Base confidence on number and quality of results
        base_confidence = min(0.9, len(results) * 0.15)
        
        # Adjust based on scores if available
        scores = [result.get('score', 0.5) for result in results]
        if scores:
            avg_score = sum(scores) / len(scores)
            base_confidence = (base_confidence + avg_score) / 2
        
        return max(0.1, min(0.95, base_confidence))
    
    def combine_retrieval_results(self, results: Dict[RetrievalType, RetrievalResult], 
                                context: MultiRetrievalContext) -> str:
        """Combine results from multiple retrievers into final context."""
        if not results:
            return ""
        
        combined_parts = []
        total_tokens = 0
        
        # Prioritize results based on query classification
        priority_order = self._determine_result_priority(context.classification, results.keys())
        
        for retrieval_type in priority_order:
            if retrieval_type in results:
                result = results[retrieval_type]
                
                # Check token budget
                if total_tokens + result.token_count <= context.token_budget:
                    header = f"=== {retrieval_type.value.upper().replace('_', ' ')} ==="
                    combined_parts.append(f"{header}\n{result.content}")
                    total_tokens += result.token_count
        
        return "\n\n".join(combined_parts)
    
    def _determine_result_priority(self, classification: Any, available_types: List[RetrievalType]) -> List[RetrievalType]:
        """Determine priority order for combining results based on query classification."""
        priority_order = []
        
        # Default priority based on query intent
        if hasattr(classification, 'intent'):
            if classification.intent.value in ['lookup', 'aggregation']:
                # Column details first for data queries
                priority_order = [RetrievalType.COLUMN_DESCRIPTIONS, RetrievalType.TABLE_DESCRIPTIONS, RetrievalType.HYBRID]
            elif classification.intent.value in ['relationship', 'analysis']:
                # Table structure first for complex queries
                priority_order = [RetrievalType.TABLE_DESCRIPTIONS, RetrievalType.COLUMN_DESCRIPTIONS, RetrievalType.HYBRID]
            else:
                # Hybrid for complex understanding
                priority_order = [RetrievalType.HYBRID, RetrievalType.COLUMN_DESCRIPTIONS, RetrievalType.TABLE_DESCRIPTIONS]
        else:
            # Default order
            priority_order = [RetrievalType.HYBRID, RetrievalType.COLUMN_DESCRIPTIONS, RetrievalType.TABLE_DESCRIPTIONS]
        
        # Filter to only available types
        return [t for t in priority_order if t in available_types]


def create_multi_retrieval_system(engine: Engine, schema_data: Optional[Dict] = None, 
                                config: Optional[Dict] = None) -> MultiRetrievalSystem:
    """Factory function to create a multi-retrieval system."""
    return MultiRetrievalSystem(engine, schema_data, config)


def intelligent_multi_retrieve(system: MultiRetrievalSystem, query: str, 
                             classification: Any, table_context: Optional[Dict] = None,
                             token_budget: int = 3000) -> str:
    """Convenience function for intelligent multi-retrieval."""
    
    # Determine which retrieval types to use based on query
    retrieval_types = []
    query_lower = query.lower()
    
    # Check if query asks for specific information types
    if any(word in query_lower for word in ['column', 'field', 'attribute', 'data type', 'values']):
        retrieval_types.append(RetrievalType.COLUMN_DESCRIPTIONS)
    
    if any(word in query_lower for word in ['table', 'schema', 'structure', 'relationship', 'entity']):
        retrieval_types.append(RetrievalType.TABLE_DESCRIPTIONS)
    
    # If no specific type detected or complex query, use hybrid
    if not retrieval_types or (hasattr(classification, 'complexity') and 
                              classification.complexity.value in ['complex', 'moderate']):
        retrieval_types.append(RetrievalType.HYBRID)
    
    # Create retrieval context
    context = MultiRetrievalContext(
        query=query,
        retrieval_types=retrieval_types,
        table_context=table_context or {},
        token_budget=token_budget,
        classification=classification
    )
    
    # Perform retrieval
    results = system.retrieve(context)
    
    # Combine results
    return system.combine_retrieval_results(results, context)
