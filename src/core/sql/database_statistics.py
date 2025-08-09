"""
Database Statistics Collection and Analysis System
This module provides comprehensive database statistics for query optimization
"""

import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from sqlalchemy import inspect, text, MetaData, Table, Column
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TableStatistics:
    """Statistics for a database table."""
    table_name: str
    row_count: int
    size_bytes: Optional[int] = None
    size_human: Optional[str] = None
    last_updated: Optional[datetime] = None
    column_count: int = 0
    index_count: int = 0
    avg_row_size: Optional[float] = None
    data_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.data_types is None:
            self.data_types = {}

@dataclass
class ColumnStatistics:
    """Statistics for a database column."""
    table_name: str
    column_name: str
    data_type: str
    null_count: int = 0
    unique_count: int = 0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_length: Optional[float] = None
    most_frequent_values: List[Tuple[Any, int]] = None
    cardinality_ratio: Optional[float] = None  # unique_count / total_count
    
    def __post_init__(self):
        if self.most_frequent_values is None:
            self.most_frequent_values = []

@dataclass
class IndexStatistics:
    """Statistics for a database index."""
    table_name: str
    index_name: str
    columns: List[str]
    is_unique: bool
    size_bytes: Optional[int] = None
    usage_count: Optional[int] = None
    selectivity: Optional[float] = None

@dataclass
class QueryPerformanceStats:
    """Query performance statistics."""
    query_hash: str
    execution_count: int
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    last_executed: datetime
    tables_accessed: Set[str]
    
    def __post_init__(self):
        if isinstance(self.tables_accessed, list):
            self.tables_accessed = set(self.tables_accessed)

class DatabaseStatisticsCollector:
    """Comprehensive database statistics collector and analyzer."""
    
    def __init__(self, engine: Engine, config: Dict[str, Any] = None):
        self.engine = engine
        self.config = config or {}
        self.stats_config = self.config.get('database', {}).get('statistics', {})
        self.inspector = inspect(engine)
        
        # Statistics cache
        self.table_stats: Dict[str, TableStatistics] = {}
        self.column_stats: Dict[str, Dict[str, ColumnStatistics]] = {}
        self.index_stats: Dict[str, List[IndexStatistics]] = {}
        self.query_stats: Dict[str, QueryPerformanceStats] = {}
        
        # Configuration
        self.cache_ttl = timedelta(hours=self.stats_config.get('cache_statistics_hours', 6))
        self.enable_row_counts = self.stats_config.get('enable_row_counts', True)
        self.enable_column_stats = self.stats_config.get('enable_column_statistics', True)
        self.enable_index_analysis = self.stats_config.get('enable_index_analysis', True)
        
        # Database-specific methods
        self.db_dialect = engine.dialect.name.lower()
        
        logger.info(f"Database statistics collector initialized for {self.db_dialect}")
    
    def collect_all_statistics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Collect comprehensive database statistics."""
        start_time = time.time()
        
        try:
            logger.info("Starting comprehensive statistics collection")
            
            # Check if we need to refresh cache
            if not force_refresh and self._is_cache_fresh():
                logger.info("Using cached statistics")
                return self._get_cached_summary()
            
            # Collect table statistics
            if self.enable_row_counts:
                self._collect_table_statistics()
            
            # Collect column statistics
            if self.enable_column_stats:
                self._collect_column_statistics()
            
            # Collect index statistics
            if self.enable_index_analysis:
                self._collect_index_statistics()
            
            collection_time = time.time() - start_time
            
            summary = {
                'collection_timestamp': datetime.now().isoformat(),
                'collection_time_seconds': round(collection_time, 2),
                'database_dialect': self.db_dialect,
                'tables_analyzed': len(self.table_stats),
                'total_rows': sum(stats.row_count for stats in self.table_stats.values()),
                'statistics_summary': self._generate_summary()
            }
            
            logger.info(f"Statistics collection completed in {collection_time:.2f} seconds")
            return summary
            
        except Exception as e:
            logger.error(f"Statistics collection failed: {e}")
            raise
    
    def _collect_table_statistics(self):
        """Collect table-level statistics."""
        logger.info("Collecting table statistics")
        
        table_names = self.inspector.get_table_names()
        
        for table_name in table_names:
            try:
                # Get row count
                row_count = self._get_row_count(table_name)
                
                # Get column information
                columns = self.inspector.get_columns(table_name)
                column_count = len(columns)
                
                # Get index information
                indexes = self.inspector.get_indexes(table_name)
                index_count = len(indexes)
                
                # Calculate data type distribution
                data_types = {}
                for col in columns:
                    dtype = str(col['type']).split('(')[0]  # Remove size specifications
                    data_types[dtype] = data_types.get(dtype, 0) + 1
                
                # Get table size (database-specific)
                size_info = self._get_table_size(table_name)
                
                stats = TableStatistics(
                    table_name=table_name,
                    row_count=row_count,
                    size_bytes=size_info.get('size_bytes'),
                    size_human=size_info.get('size_human'),
                    last_updated=datetime.now(),
                    column_count=column_count,
                    index_count=index_count,
                    avg_row_size=size_info.get('avg_row_size'),
                    data_types=data_types
                )
                
                self.table_stats[table_name] = stats
                
                logger.debug(f"Table {table_name}: {row_count:,} rows, {column_count} columns")
                
            except Exception as e:
                logger.warning(f"Failed to collect statistics for table {table_name}: {e}")
    
    def _collect_column_statistics(self):
        """Collect column-level statistics."""
        logger.info("Collecting column statistics")
        
        for table_name in self.table_stats.keys():
            try:
                self.column_stats[table_name] = {}
                columns = self.inspector.get_columns(table_name)
                total_rows = self.table_stats[table_name].row_count
                
                if total_rows == 0:
                    continue
                
                for column in columns:
                    col_name = column['name']
                    col_type = str(column['type'])
                    
                    try:
                        # Collect column statistics
                        col_stats = self._analyze_column(table_name, col_name, col_type, total_rows)
                        self.column_stats[table_name][col_name] = col_stats
                        
                    except Exception as e:
                        logger.debug(f"Failed to analyze column {table_name}.{col_name}: {e}")
                
                logger.debug(f"Analyzed {len(self.column_stats[table_name])} columns in {table_name}")
                
            except Exception as e:
                logger.warning(f"Failed to collect column statistics for {table_name}: {e}")
    
    def _collect_index_statistics(self):
        """Collect index statistics."""
        logger.info("Collecting index statistics")
        
        for table_name in self.table_stats.keys():
            try:
                indexes = self.inspector.get_indexes(table_name)
                self.index_stats[table_name] = []
                
                for index in indexes:
                    index_name = index['name']
                    columns = index['column_names']
                    is_unique = index.get('unique', False)
                    
                    # Calculate index selectivity
                    selectivity = self._calculate_index_selectivity(table_name, columns)
                    
                    # Get index size (database-specific)
                    size_bytes = self._get_index_size(table_name, index_name)
                    
                    index_stats = IndexStatistics(
                        table_name=table_name,
                        index_name=index_name,
                        columns=columns,
                        is_unique=is_unique,
                        size_bytes=size_bytes,
                        selectivity=selectivity
                    )
                    
                    self.index_stats[table_name].append(index_stats)
                
                logger.debug(f"Analyzed {len(indexes)} indexes for {table_name}")
                
            except Exception as e:
                logger.warning(f"Failed to collect index statistics for {table_name}: {e}")
    
    def _get_row_count(self, table_name: str) -> int:
        """Get accurate row count for table."""
        try:
            # Use COUNT(*) for accurate count
            query = text(f"SELECT COUNT(*) FROM {self._quote_identifier(table_name)}")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return result.scalar()
                
        except Exception as e:
            logger.debug(f"COUNT query failed for {table_name}, trying alternative: {e}")
            
            # Fallback to database-specific row count estimates
            return self._get_estimated_row_count(table_name)
    
    def _get_estimated_row_count(self, table_name: str) -> int:
        """Get estimated row count using database-specific methods."""
        try:
            if self.db_dialect == 'postgresql':
                query = text("""
                    SELECT n_tup_ins - n_tup_del as row_count
                    FROM pg_stat_user_tables
                    WHERE relname = :table_name
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {'table_name': table_name})
                    row = result.fetchone()
                    return row[0] if row and row[0] is not None else 0
            
            elif self.db_dialect == 'mysql':
                query = text("""
                    SELECT table_rows
                    FROM information_schema.tables
                    WHERE table_name = :table_name
                    AND table_schema = DATABASE()
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {'table_name': table_name})
                    row = result.fetchone()
                    return row[0] if row and row[0] is not None else 0
            
            elif self.db_dialect == 'sqlite':
                # SQLite doesn't have built-in row count statistics
                return 0
            
        except Exception as e:
            logger.debug(f"Estimated row count failed for {table_name}: {e}")
        
        return 0
    
    def _get_table_size(self, table_name: str) -> Dict[str, Any]:
        """Get table size information."""
        size_info = {}
        
        try:
            if self.db_dialect == 'postgresql':
                query = text("""
                    SELECT 
                        pg_total_relation_size(:table_name) as total_bytes,
                        pg_size_pretty(pg_total_relation_size(:table_name)) as size_pretty,
                        pg_relation_size(:table_name) / GREATEST(
                            (SELECT n_tup_ins - n_tup_del FROM pg_stat_user_tables WHERE relname = :table_name), 
                            1
                        ) as avg_row_size
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {'table_name': table_name})
                    row = result.fetchone()
                    if row:
                        size_info = {
                            'size_bytes': row[0],
                            'size_human': row[1],
                            'avg_row_size': row[2]
                        }
            
            elif self.db_dialect == 'mysql':
                query = text("""
                    SELECT 
                        data_length + index_length as total_bytes,
                        ROUND((data_length + index_length) / 1024 / 1024, 2) as size_mb,
                        avg_row_length
                    FROM information_schema.tables
                    WHERE table_name = :table_name
                    AND table_schema = DATABASE()
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {'table_name': table_name})
                    row = result.fetchone()
                    if row:
                        size_info = {
                            'size_bytes': row[0],
                            'size_human': f"{row[1]} MB",
                            'avg_row_size': row[2]
                        }
            
        except Exception as e:
            logger.debug(f"Table size calculation failed for {table_name}: {e}")
        
        return size_info
    
    def _analyze_column(self, table_name: str, column_name: str, column_type: str, 
                       total_rows: int) -> ColumnStatistics:
        """Analyze individual column statistics."""
        
        quoted_table = self._quote_identifier(table_name)
        quoted_column = self._quote_identifier(column_name)
        
        # Basic statistics query
        stats_query = text(f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT({quoted_column}) as non_null_count,
                COUNT(DISTINCT {quoted_column}) as unique_count
            FROM {quoted_table}
        """)
        
        col_stats = ColumnStatistics(
            table_name=table_name,
            column_name=column_name,
            data_type=column_type
        )
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(stats_query)
                row = result.fetchone()
                
                if row:
                    total_count = row[0]
                    non_null_count = row[1]
                    unique_count = row[2]
                    
                    col_stats.null_count = total_count - non_null_count
                    col_stats.unique_count = unique_count
                    
                    if total_count > 0:
                        col_stats.cardinality_ratio = unique_count / total_count
                
                # Get min/max for numeric and date columns
                if self._is_numeric_type(column_type) or self._is_date_type(column_type):
                    minmax_query = text(f"""
                        SELECT MIN({quoted_column}), MAX({quoted_column})
                        FROM {quoted_table}
                        WHERE {quoted_column} IS NOT NULL
                    """)
                    
                    result = conn.execute(minmax_query)
                    row = result.fetchone()
                    if row:
                        col_stats.min_value = row[0]
                        col_stats.max_value = row[1]
                
                # Get average length for string columns
                if self._is_string_type(column_type):
                    length_query = text(f"""
                        SELECT AVG(LENGTH({quoted_column}))
                        FROM {quoted_table}
                        WHERE {quoted_column} IS NOT NULL
                    """)
                    
                    try:
                        result = conn.execute(length_query)
                        row = result.fetchone()
                        if row and row[0] is not None:
                            col_stats.avg_length = float(row[0])
                    except Exception as e:
                        logger.debug(f"Length calculation failed for {column_name}: {e}")
                
                # Get most frequent values (top 5)
                if unique_count < total_rows * 0.8:  # Only if not too unique
                    freq_query = text(f"""
                        SELECT {quoted_column}, COUNT(*) as freq
                        FROM {quoted_table}
                        WHERE {quoted_column} IS NOT NULL
                        GROUP BY {quoted_column}
                        ORDER BY freq DESC
                        LIMIT 5
                    """)
                    
                    try:
                        result = conn.execute(freq_query)
                        col_stats.most_frequent_values = [(row[0], row[1]) for row in result]
                    except Exception as e:
                        logger.debug(f"Frequency calculation failed for {column_name}: {e}")
        
        except Exception as e:
            logger.debug(f"Column analysis failed for {table_name}.{column_name}: {e}")
        
        return col_stats
    
    def _calculate_index_selectivity(self, table_name: str, columns: List[str]) -> Optional[float]:
        """Calculate index selectivity."""
        try:
            if not columns:
                return None
            
            quoted_table = self._quote_identifier(table_name)
            quoted_columns = [self._quote_identifier(col) for col in columns]
            
            # Calculate selectivity as unique_combinations / total_rows
            selectivity_query = text(f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT {', '.join(quoted_columns)}) as unique_combinations
                FROM {quoted_table}
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(selectivity_query)
                row = result.fetchone()
                
                if row and row[0] > 0:
                    return row[1] / row[0]
                    
        except Exception as e:
            logger.debug(f"Index selectivity calculation failed for {table_name}.{columns}: {e}")
        
        return None
    
    def _get_index_size(self, table_name: str, index_name: str) -> Optional[int]:
        """Get index size in bytes."""
        try:
            if self.db_dialect == 'postgresql':
                query = text("""
                    SELECT pg_relation_size(indexrelid)
                    FROM pg_stat_user_indexes
                    WHERE relname = :table_name AND indexrelname = :index_name
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {
                        'table_name': table_name,
                        'index_name': index_name
                    })
                    row = result.fetchone()
                    return row[0] if row else None
                    
        except Exception as e:
            logger.debug(f"Index size calculation failed for {table_name}.{index_name}: {e}")
        
        return None
    
    def _is_cache_fresh(self) -> bool:
        """Check if statistics cache is still fresh."""
        if not self.table_stats:
            return False
        
        # Check if any table stats are older than cache TTL
        now = datetime.now()
        for stats in self.table_stats.values():
            if stats.last_updated and (now - stats.last_updated) > self.cache_ttl:
                return False
        
        return True
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive statistics summary."""
        summary = {
            'tables': {
                'total_count': len(self.table_stats),
                'total_rows': sum(stats.row_count for stats in self.table_stats.values()),
                'largest_table': None,
                'table_sizes': {}
            },
            'columns': {
                'total_count': sum(len(cols) for cols in self.column_stats.values()),
                'data_type_distribution': defaultdict(int),
                'highly_unique_columns': [],
                'columns_with_nulls': []
            },
            'indexes': {
                'total_count': sum(len(indexes) for indexes in self.index_stats.values()),
                'unique_indexes': 0,
                'low_selectivity_indexes': []
            }
        }
        
        # Analyze tables
        if self.table_stats:
            largest_table = max(self.table_stats.values(), key=lambda x: x.row_count)
            summary['tables']['largest_table'] = {
                'name': largest_table.table_name,
                'rows': largest_table.row_count
            }
            
            for name, stats in self.table_stats.items():
                summary['tables']['table_sizes'][name] = {
                    'rows': stats.row_count,
                    'size_human': stats.size_human
                }
        
        # Analyze columns
        for table_cols in self.column_stats.values():
            for col_stats in table_cols.values():
                # Data type distribution
                base_type = col_stats.data_type.split('(')[0]
                summary['columns']['data_type_distribution'][base_type] += 1
                
                # Highly unique columns (cardinality > 0.8)
                if col_stats.cardinality_ratio and col_stats.cardinality_ratio > 0.8:
                    summary['columns']['highly_unique_columns'].append(
                        f"{col_stats.table_name}.{col_stats.column_name}"
                    )
                
                # Columns with significant nulls (> 10%)
                total_rows = self.table_stats.get(col_stats.table_name, TableStatistics('', 0)).row_count
                if total_rows > 0 and (col_stats.null_count / total_rows) > 0.1:
                    summary['columns']['columns_with_nulls'].append(
                        f"{col_stats.table_name}.{col_stats.column_name}"
                    )
        
        # Analyze indexes
        for table_indexes in self.index_stats.values():
            for index_stats in table_indexes:
                if index_stats.is_unique:
                    summary['indexes']['unique_indexes'] += 1
                
                # Low selectivity indexes (< 0.1)
                if index_stats.selectivity and index_stats.selectivity < 0.1:
                    summary['indexes']['low_selectivity_indexes'].append(
                        f"{index_stats.table_name}.{index_stats.index_name}"
                    )
        
        return summary
    
    def _get_cached_summary(self) -> Dict[str, Any]:
        """Get summary from cached statistics."""
        return {
            'collection_timestamp': 'cached',
            'database_dialect': self.db_dialect,
            'tables_analyzed': len(self.table_stats),
            'total_rows': sum(stats.row_count for stats in self.table_stats.values()),
            'statistics_summary': self._generate_summary(),
            'cache_status': 'fresh'
        }
    
    def _quote_identifier(self, identifier: str) -> str:
        """Quote database identifier based on dialect."""
        if self.db_dialect == 'postgresql':
            return f'"{identifier}"'
        elif self.db_dialect == 'mysql':
            return f'`{identifier}`'
        elif self.db_dialect == 'sqlite':
            return f'"{identifier}"'
        else:
            return identifier
    
    def _is_numeric_type(self, column_type: str) -> bool:
        """Check if column type is numeric."""
        numeric_types = ['INTEGER', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'DOUBLE', 'BIGINT', 'SMALLINT']
        return any(ntype in column_type.upper() for ntype in numeric_types)
    
    def _is_string_type(self, column_type: str) -> bool:
        """Check if column type is string."""
        string_types = ['VARCHAR', 'CHAR', 'TEXT', 'STRING', 'CLOB']
        return any(stype in column_type.upper() for stype in string_types)
    
    def _is_date_type(self, column_type: str) -> bool:
        """Check if column type is date/time."""
        date_types = ['DATE', 'TIME', 'TIMESTAMP', 'DATETIME']
        return any(dtype in column_type.upper() for dtype in date_types)
    
    def get_table_statistics(self, table_name: str) -> Optional[TableStatistics]:
        """Get statistics for a specific table."""
        return self.table_stats.get(table_name)
    
    def get_column_statistics(self, table_name: str, column_name: str) -> Optional[ColumnStatistics]:
        """Get statistics for a specific column."""
        return self.column_stats.get(table_name, {}).get(column_name)
    
    def get_optimization_insights(self) -> Dict[str, List[str]]:
        """Get optimization insights based on collected statistics."""
        insights = {
            'performance_recommendations': [],
            'index_recommendations': [],
            'schema_recommendations': [],
            'query_recommendations': []
        }
        
        # Performance recommendations
        for stats in self.table_stats.values():
            if stats.row_count > 1000000:  # Large tables
                insights['performance_recommendations'].append(
                    f"Table {stats.table_name} has {stats.row_count:,} rows - consider partitioning"
                )
        
        # Index recommendations
        for table_name, indexes in self.index_stats.items():
            for index in indexes:
                if index.selectivity and index.selectivity < 0.05:
                    insights['index_recommendations'].append(
                        f"Index {index.index_name} on {table_name} has low selectivity ({index.selectivity:.3f})"
                    )
        
        # Schema recommendations
        for table_name, columns in self.column_stats.items():
            for col_stats in columns.values():
                if col_stats.null_count > 0:
                    total_rows = self.table_stats[table_name].row_count
                    null_percentage = (col_stats.null_count / total_rows) * 100
                    if null_percentage > 50:
                        insights['schema_recommendations'].append(
                            f"Column {table_name}.{col_stats.column_name} is {null_percentage:.1f}% null"
                        )
        
        return insights
    
    def export_statistics(self, format: str = 'json') -> str:
        """Export collected statistics in specified format."""
        stats_data = {
            'collection_info': {
                'timestamp': datetime.now().isoformat(),
                'database_dialect': self.db_dialect,
                'tables_count': len(self.table_stats)
            },
            'table_statistics': {name: asdict(stats) for name, stats in self.table_stats.items()},
            'column_statistics': {
                table: {col: asdict(stats) for col, stats in cols.items()}
                for table, cols in self.column_stats.items()
            },
            'index_statistics': {
                table: [asdict(stats) for stats in indexes]
                for table, indexes in self.index_stats.items()
            }
        }
        
        if format.lower() == 'json':
            return json.dumps(stats_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global statistics collector instance
_global_stats_collector = None

def get_database_statistics(engine: Engine, config: Dict[str, Any] = None) -> DatabaseStatisticsCollector:
    """Get global database statistics collector."""
    global _global_stats_collector
    
    if _global_stats_collector is None or _global_stats_collector.engine != engine:
        _global_stats_collector = DatabaseStatisticsCollector(engine, config)
    
    return _global_stats_collector

def reset_database_statistics():
    """Reset global database statistics collector."""
    global _global_stats_collector
    _global_stats_collector = None
