
import hashlib
import json
import sqlite3
import pickle
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryCacheEntry:
    """Represents a cached query with metadata."""
    query_hash: str
    original_query: str
    normalized_query: str
    sql_query: str
    result_hash: str
    result_data: Any
    classification: Dict[str, Any]
    context_hash: str
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    cost_saved: float
    confidence_score: float

@dataclass
class CostMetrics:
    """Cost tracking and optimization metrics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    average_response_time: float = 0.0
    model_usage_breakdown: Dict[str, int] = None
    
    def __post_init__(self):
        if self.model_usage_breakdown is None:
            self.model_usage_breakdown = defaultdict(int)

class IntelligentCostOptimizer:
    """Advanced cost optimization system with intelligent caching and routing."""
    
    def __init__(self, cache_dir: str = "cache", config: Dict = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.config = config or {}
        self.optimization_config = self.config.get('cost_optimization', {})
        
        # Cache configuration
        self.max_cache_size = self.optimization_config.get('max_cache_entries', 1000)
        self.cache_ttl_hours = self.optimization_config.get('cache_ttl_hours', 24)
        self.similarity_threshold = self.optimization_config.get('similarity_threshold', 0.88)
        self.enable_result_caching = self.optimization_config.get('enable_result_caching', True)
        self.enable_context_caching = self.optimization_config.get('enable_context_caching', True)
        
        # Initialize cache storage
        self.cache_db_path = self.cache_dir / "query_cache.db"
        self.context_cache_path = self.cache_dir / "context_cache.pkl"
        self.metrics_path = self.cache_dir / "cost_metrics.json"
        
        # In-memory caches
        self.query_cache: Dict[str, QueryCacheEntry] = {}
        self.context_cache: Dict[str, Dict] = {}
        self.similar_queries: Dict[str, List[str]] = defaultdict(list)
        
        # Metrics tracking
        self.metrics = CostMetrics()
        
        # Initialize storage
        self._initialize_storage()
        self._load_cache()
        self._load_metrics()
        
    def _initialize_storage(self):
        """Initialize SQLite database for persistent caching."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        query_hash TEXT PRIMARY KEY,
                        original_query TEXT NOT NULL,
                        normalized_query TEXT NOT NULL,
                        sql_query TEXT,
                        result_hash TEXT,
                        result_data BLOB,
                        classification TEXT,
                        context_hash TEXT,
                        timestamp DATETIME,
                        access_count INTEGER DEFAULT 1,
                        last_accessed DATETIME,
                        cost_saved REAL DEFAULT 0.0,
                        confidence_score REAL DEFAULT 0.8
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON query_cache(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_normalized_query ON query_cache(normalized_query)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_access_count ON query_cache(access_count)
                """)
                
        except Exception as e:
            logger.error(f"Failed to initialize cache storage: {e}")
    
    def _load_cache(self):
        """Load cache from persistent storage."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM query_cache 
                    WHERE datetime(timestamp) > datetime('now', '-{} hours')
                    ORDER BY access_count DESC, last_accessed DESC
                    LIMIT {}
                """.format(self.cache_ttl_hours, self.max_cache_size))
                
                for row in cursor.fetchall():
                    entry = QueryCacheEntry(
                        query_hash=row[0],
                        original_query=row[1],
                        normalized_query=row[2],
                        sql_query=row[3],
                        result_hash=row[4],
                        result_data=pickle.loads(row[5]) if row[5] else None,
                        classification=json.loads(row[6]) if row[6] else {},
                        context_hash=row[7],
                        timestamp=datetime.fromisoformat(row[8]),
                        access_count=row[9],
                        last_accessed=datetime.fromisoformat(row[10]),
                        cost_saved=row[11],
                        confidence_score=row[12]
                    )
                    self.query_cache[entry.query_hash] = entry
            
            # Load context cache if it exists
            if self.context_cache_path.exists():
                with open(self.context_cache_path, 'rb') as f:
                    self.context_cache = pickle.load(f)
                    
            logger.info(f"Loaded {len(self.query_cache)} cached queries and {len(self.context_cache)} contexts")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def _load_metrics(self):
        """Load cost metrics from persistent storage."""
        try:
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = CostMetrics(**metrics_data)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def _save_metrics(self):
        """Save cost metrics to persistent storage."""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better cache matching."""
        import re
        
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Normalize common variations
        replacements = [
            (r'\bshow me\b', 'show'),
            (r'\bget me\b', 'get'),
            (r'\btell me\b', 'tell'),
            (r'\bi want to\b', ''),
            (r'\bcan you\b', ''),
            (r'\bplease\b', ''),
            (r'\bhow many\b', 'count'),
            (r'\btotal number of\b', 'count'),
            (r'\bnumber of\b', 'count'),
        ]
        
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove punctuation except essential ones
        normalized = re.sub(r'[^\w\s\-\']', '', normalized)
        
        # Remove extra spaces again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        try:
            # Normalize both queries
            norm_query1 = self._normalize_query(query1)
            norm_query2 = self._normalize_query(query2)
            
            # If normalized queries are identical, return 1.0
            if norm_query1 == norm_query2:
                return 1.0
            
            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            documents = [norm_query1, norm_query2]
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            # Fallback to simple word overlap
            words1 = set(query1.lower().split())
            words2 = set(query2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
    
    def _generate_cache_key(self, query: str, context_hash: str = "", classification: Dict = None) -> str:
        """Generate unique cache key for query."""
        key_components = [
            self._normalize_query(query),
            context_hash,
            json.dumps(classification or {}, sort_keys=True)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_result_hash(self, result: Any) -> str:
        """Generate hash for result data."""
        try:
            if isinstance(result, pd.DataFrame):
                # Hash based on shape and first few rows
                content = f"{result.shape}_{result.head().to_string()}"
            else:
                content = str(result)
            
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except:
            return "unknown_result"
    
    def find_similar_cached_query(self, query: str, context_hash: str = "", classification: Dict = None) -> Optional[QueryCacheEntry]:
        """Find similar cached query that can be reused."""
        if not self.enable_result_caching or not self.query_cache:
            return None
        
        normalized_query = self._normalize_query(query)
        best_match = None
        best_similarity = 0.0
        
        # First, try exact match on normalized query
        for entry in self.query_cache.values():
            if entry.normalized_query == normalized_query:
                # Update access metrics
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self.metrics.cache_hits += 1
                
                logger.info(f"Exact cache hit for query: {query[:50]}...")
                return entry
        
        # Then try semantic similarity matching
        for entry in self.query_cache.values():
            similarity = self._calculate_query_similarity(query, entry.original_query)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                # Additional context matching for high-similarity queries
                context_match = True
                if context_hash and entry.context_hash:
                    context_match = context_hash == entry.context_hash or similarity > 0.95
                
                if context_match:
                    best_similarity = similarity
                    best_match = entry
        
        if best_match and best_similarity >= self.similarity_threshold:
            # Update access metrics
            best_match.access_count += 1
            best_match.last_accessed = datetime.now()
            self.metrics.cache_hits += 1
            
            logger.info(f"Semantic cache hit (similarity: {best_similarity:.3f}) for query: {query[:50]}...")
            return best_match
        
        # No suitable match found
        self.metrics.cache_misses += 1
        return None
    
    def cache_query_result(self, query: str, sql_query: str, result: Any, 
                          classification: Dict, context_hash: str = "", 
                          cost_saved: float = 0.0, confidence_score: float = 0.8):
        """Cache query result with intelligent storage."""
        if not self.enable_result_caching:
            return
        
        try:
            query_hash = self._generate_cache_key(query, context_hash, classification)
            result_hash = self._generate_result_hash(result)
            normalized_query = self._normalize_query(query)
            
            # Create cache entry
            entry = QueryCacheEntry(
                query_hash=query_hash,
                original_query=query,
                normalized_query=normalized_query,
                sql_query=sql_query,
                result_hash=result_hash,
                result_data=result,
                classification=classification,
                context_hash=context_hash,
                timestamp=datetime.now(),
                access_count=1,
                last_accessed=datetime.now(),
                cost_saved=cost_saved,
                confidence_score=confidence_score
            )
            
            # Store in memory cache
            self.query_cache[query_hash] = entry
            
            # Persist to database
            self._persist_cache_entry(entry)
            
            # Clean up old entries if cache is too large
            if len(self.query_cache) > self.max_cache_size:
                self._cleanup_cache()
            
            logger.debug(f"Cached query result: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to cache query result: {e}")
    
    def _persist_cache_entry(self, entry: QueryCacheEntry):
        """Persist cache entry to database."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO query_cache 
                    (query_hash, original_query, normalized_query, sql_query, 
                     result_hash, result_data, classification, context_hash,
                     timestamp, access_count, last_accessed, cost_saved, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.query_hash,
                    entry.original_query,
                    entry.normalized_query,
                    entry.sql_query,
                    entry.result_hash,
                    pickle.dumps(entry.result_data) if entry.result_data is not None else None,
                    json.dumps(entry.classification),
                    entry.context_hash,
                    entry.timestamp.isoformat(),
                    entry.access_count,
                    entry.last_accessed.isoformat(),
                    entry.cost_saved,
                    entry.confidence_score
                ))
        except Exception as e:
            logger.error(f"Failed to persist cache entry: {e}")
    
    def _cleanup_cache(self):
        """Clean up old and less frequently used cache entries."""
        try:
            # Sort by access frequency and recency
            sorted_entries = sorted(
                self.query_cache.values(),
                key=lambda e: (e.access_count, e.last_accessed),
                reverse=True
            )
            
            # Keep top entries
            keep_count = int(self.max_cache_size * 0.8)  # Keep 80% of max
            entries_to_keep = sorted_entries[:keep_count]
            entries_to_remove = sorted_entries[keep_count:]
            
            # Update memory cache
            self.query_cache = {e.query_hash: e for e in entries_to_keep}
            
            # Remove from database
            if entries_to_remove:
                hashes_to_remove = [e.query_hash for e in entries_to_remove]
                with sqlite3.connect(self.cache_db_path) as conn:
                    placeholders = ','.join(['?'] * len(hashes_to_remove))
                    conn.execute(f"""
                        DELETE FROM query_cache 
                        WHERE query_hash IN ({placeholders})
                    """, hashes_to_remove)
                
                logger.info(f"Cleaned up {len(entries_to_remove)} cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def get_cached_context(self, context_key: str) -> Optional[str]:
        """Get cached RAG context."""
        if not self.enable_context_caching:
            return None
        
        cached_context = self.context_cache.get(context_key)
        if cached_context:
            # Check if context is still fresh
            cache_time = cached_context.get('timestamp', datetime.min)
            if isinstance(cache_time, str):
                cache_time = datetime.fromisoformat(cache_time)
            
            if datetime.now() - cache_time < timedelta(hours=self.cache_ttl_hours):
                return cached_context.get('content')
        
        return None
    
    def cache_context(self, context_key: str, content: str):
        """Cache RAG context with timestamp."""
        if not self.enable_context_caching:
            return
        
        self.context_cache[context_key] = {
            'content': content,
            'timestamp': datetime.now(),
            'access_count': 1
        }
        
        # Persist context cache
        try:
            with open(self.context_cache_path, 'wb') as f:
                pickle.dump(self.context_cache, f)
        except Exception as e:
            logger.error(f"Failed to persist context cache: {e}")
    
    def estimate_cost_savings(self, query: str, classification: Dict) -> float:
        """Estimate potential cost savings from caching."""
        # Base cost estimates (in USD)
        model_costs = {
            'gpt-4o': {'input': 5.0 / 1_000_000, 'output': 15.0 / 1_000_000},
            'gpt-4o-mini': {'input': 0.15 / 1_000_000, 'output': 0.6 / 1_000_000}
        }
        
        # Estimate tokens based on query complexity
        complexity = classification.get('complexity', 'moderate')
        token_estimates = {
            'simple': 800,
            'moderate': 1500,
            'complex': 3000
        }
        
        estimated_tokens = token_estimates.get(complexity, 1500)
        model = classification.get('recommended_model', 'gpt-4o-mini')
        
        if model in model_costs:
            # Assume 70% input, 30% output tokens
            input_tokens = int(estimated_tokens * 0.7)
            output_tokens = int(estimated_tokens * 0.3)
            
            cost = (input_tokens * model_costs[model]['input'] + 
                   output_tokens * model_costs[model]['output'])
            
            return cost
        
        return 0.0
    
    def update_metrics(self, query_cost: float, response_time: float, model_used: str, tokens_used: int):
        """Update cost optimization metrics."""
        self.metrics.total_queries += 1
        self.metrics.total_cost_usd += query_cost
        self.metrics.total_tokens_used += tokens_used
        self.metrics.model_usage_breakdown[model_used] += 1
        
        # Update average response time
        if self.metrics.total_queries == 1:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_queries - 1) + response_time) 
                / self.metrics.total_queries
            )
        
        # Save metrics periodically
        if self.metrics.total_queries % 10 == 0:
            self._save_metrics()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive cost optimization report."""
        cache_hit_rate = (
            self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
        ) * 100
        
        return {
            "cache_performance": {
                "total_entries": len(self.query_cache),
                "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cost_saved_usd": f"${self.metrics.cost_saved_usd:.4f}"
            },
            "cost_metrics": {
                "total_queries": self.metrics.total_queries,
                "total_cost_usd": f"${self.metrics.total_cost_usd:.4f}",
                "average_cost_per_query": f"${self.metrics.total_cost_usd / max(1, self.metrics.total_queries):.4f}",
                "total_tokens": self.metrics.total_tokens_used,
                "average_tokens_per_query": int(self.metrics.total_tokens_used / max(1, self.metrics.total_queries))
            },
            "performance_metrics": {
                "average_response_time": f"{self.metrics.average_response_time:.2f}s",
                "model_usage": dict(self.metrics.model_usage_breakdown)
            },
            "optimization_recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on usage patterns."""
        recommendations = []
        
        cache_hit_rate = (
            self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
        ) * 100
        
        if cache_hit_rate < 20:
            recommendations.append("Consider increasing cache TTL or similarity threshold for better cache utilization")
        
        if self.metrics.model_usage_breakdown.get('gpt-4o', 0) > self.metrics.model_usage_breakdown.get('gpt-4o-mini', 0):
            recommendations.append("High usage of gpt-4o detected. Review query classification to optimize costs")
        
        avg_tokens = self.metrics.total_tokens_used / max(1, self.metrics.total_queries)
        if avg_tokens > 2000:
            recommendations.append("High average token usage. Consider context optimization and prompt engineering")
        
        if not recommendations:
            recommendations.append("System is well-optimized. Continue monitoring for further improvements")
        
        return recommendations
    
    def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            expiry_time = datetime.now() - timedelta(hours=self.cache_ttl_hours)
            
            # Clean memory cache
            expired_keys = [
                key for key, entry in self.query_cache.items()
                if entry.timestamp < expiry_time
            ]
            
            for key in expired_keys:
                del self.query_cache[key]
            
            # Clean database
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    DELETE FROM query_cache 
                    WHERE datetime(timestamp) < datetime('now', '-{} hours')
                """.format(self.cache_ttl_hours))
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")

# Global optimizer instance
_cost_optimizer = None

def get_cost_optimizer(config: Dict = None) -> IntelligentCostOptimizer:
    """Get or create global cost optimizer instance."""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = IntelligentCostOptimizer(config=config)
    return _cost_optimizer

def reset_cost_optimizer(config: Dict = None) -> IntelligentCostOptimizer:
    """Reset cost optimizer instance."""
    global _cost_optimizer
    if _cost_optimizer:
        _cost_optimizer._save_metrics()
    _cost_optimizer = IntelligentCostOptimizer(config=config)
    return _cost_optimizer