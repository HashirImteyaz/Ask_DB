"""
Advanced Multi-Level Redis Caching System for NLQ Intelligence
This module provides comprehensive caching for SQL results, LLM responses, and RAG contexts
"""

import os
import json
import hashlib
import pickle
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path

# Redis imports with fallback
try:
    import redis
    from redis.exceptions import ConnectionError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    average_retrieval_time: float = 0.0
    cache_layers: Dict[str, int] = None
    
    def __post_init__(self):
        if self.cache_layers is None:
            self.cache_layers = {}
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def add_hit(self, layer: str = "default", size_bytes: int = 0, retrieval_time: float = 0.0):
        """Record a cache hit."""
        self.hits += 1
        self.total_requests += 1
        self.total_size_bytes += size_bytes
        self.cache_layers[layer] = self.cache_layers.get(layer, 0) + 1
        
        # Update average retrieval time
        if self.total_requests > 1:
            self.average_retrieval_time = (
                (self.average_retrieval_time * (self.total_requests - 1) + retrieval_time) 
                / self.total_requests
            )
        else:
            self.average_retrieval_time = retrieval_time
    
    def add_miss(self, layer: str = "default"):
        """Record a cache miss."""
        self.misses += 1
        self.total_requests += 1

class RedisCache:
    """Advanced Redis caching system with intelligent key management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cache_config = self.config.get('cost_optimization', {})
        
        # Redis connection configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = None
        self.fallback_cache = {}  # In-memory fallback
        self.use_redis = REDIS_AVAILABLE
        
        # Cache configuration
        self.default_ttl = self.cache_config.get('cache_ttl_hours', 24) * 3600
        self.max_cache_size = self.cache_config.get('max_cache_entries', 10000)
        self.compression_enabled = self.cache_config.get('enable_context_compression', True)
        
        # Cache metrics
        self.metrics = CacheMetrics()
        
        # Initialize Redis connection
        self._initialize_redis()
        
        # Cache key prefixes for different layers
        self.key_prefixes = {
            'sql_results': 'nlq:sql:',
            'llm_responses': 'nlq:llm:',
            'rag_contexts': 'nlq:rag:',
            'query_classifications': 'nlq:classify:',
            'final_responses': 'nlq:final:',
            'user_sessions': 'nlq:session:',
            'system_stats': 'nlq:stats:'
        }
    
    def _initialize_redis(self):
        """Initialize Redis connection with error handling."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory fallback cache")
            return
        
        try:
            # Parse Redis URL and create client
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We'll handle encoding manually
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis cache initialized successfully: {self.redis_url}")
            
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            self.use_redis = False
            self.redis_client = None
    
    def _generate_cache_key(self, layer: str, identifier: str, context: Dict = None) -> str:
        """Generate consistent cache key."""
        prefix = self.key_prefixes.get(layer, 'nlq:general:')
        
        # Include context hash if provided
        if context:
            context_str = json.dumps(context, sort_keys=True)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            identifier = f"{identifier}:{context_hash}"
        
        # Create final key with hash for consistency
        key_content = f"{prefix}{identifier}"
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        
        return f"{prefix}{key_hash}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage with compression."""
        try:
            if isinstance(data, pd.DataFrame):
                # Special handling for DataFrames
                serialized = {
                    'type': 'dataframe',
                    'data': data.to_json(orient='split', date_format='iso'),
                    'dtypes': data.dtypes.to_dict()
                }
            else:
                serialized = {'type': 'general', 'data': data}
            
            # Use pickle for serialization
            raw_data = pickle.dumps(serialized)
            
            # Apply compression if enabled and data is large
            if self.compression_enabled and len(raw_data) > 1024:
                try:
                    import gzip
                    compressed_data = gzip.compress(raw_data)
                    if len(compressed_data) < len(raw_data) * 0.8:  # Only use if >20% compression
                        return b'compressed:' + compressed_data
                except ImportError:
                    pass
            
            return raw_data
            
        except Exception as e:
            logger.error(f"Data serialization failed: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            # Check if data is compressed
            if data.startswith(b'compressed:'):
                import gzip
                data = gzip.decompress(data[11:])  # Remove 'compressed:' prefix
            
            # Deserialize with pickle
            serialized = pickle.loads(data)
            
            if serialized.get('type') == 'dataframe':
                # Reconstruct DataFrame
                df = pd.read_json(serialized['data'], orient='split')
                # Restore dtypes if available
                if 'dtypes' in serialized:
                    for col, dtype in serialized['dtypes'].items():
                        try:
                            df[col] = df[col].astype(dtype)
                        except:
                            pass
                return df
            else:
                return serialized['data']
                
        except Exception as e:
            logger.error(f"Data deserialization failed: {e}")
            raise
    
    def get(self, layer: str, identifier: str, context: Dict = None) -> Optional[Any]:
        """Get data from cache with performance tracking."""
        start_time = time.time()
        cache_key = self._generate_cache_key(layer, identifier, context)
        
        try:
            if self.use_redis and self.redis_client:
                # Try Redis first
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = self._deserialize_data(cached_data)
                    retrieval_time = time.time() - start_time
                    self.metrics.add_hit(layer, len(cached_data), retrieval_time)
                    return result
            
            # Fallback to in-memory cache
            if cache_key in self.fallback_cache:
                entry = self.fallback_cache[cache_key]
                
                # Check if entry has expired
                if entry['expires_at'] > datetime.now():
                    retrieval_time = time.time() - start_time
                    self.metrics.add_hit(f"{layer}_memory", 0, retrieval_time)
                    return entry['data']
                else:
                    # Remove expired entry
                    del self.fallback_cache[cache_key]
            
            # Cache miss
            self.metrics.add_miss(layer)
            return None
            
        except Exception as e:
            logger.error(f"Cache get operation failed for {layer}: {e}")
            self.metrics.add_miss(layer)
            return None
    
    def set(self, layer: str, identifier: str, data: Any, ttl: Optional[int] = None, 
            context: Dict = None) -> bool:
        """Set data in cache with automatic management."""
        cache_key = self._generate_cache_key(layer, identifier, context)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_data = self._serialize_data(data)
            
            if self.use_redis and self.redis_client:
                # Store in Redis with TTL
                success = self.redis_client.setex(cache_key, ttl, serialized_data)
                if success:
                    return True
            
            # Store in fallback cache
            self.fallback_cache[cache_key] = {
                'data': data,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=ttl),
                'size_bytes': len(serialized_data)
            }
            
            # Clean up fallback cache if it's getting too large
            self._cleanup_fallback_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set operation failed for {layer}: {e}")
            return False
    
    def delete(self, layer: str, identifier: str, context: Dict = None) -> bool:
        """Delete specific cache entry."""
        cache_key = self._generate_cache_key(layer, identifier, context)
        
        try:
            deleted_count = 0
            
            if self.use_redis and self.redis_client:
                deleted_count += self.redis_client.delete(cache_key)
            
            if cache_key in self.fallback_cache:
                del self.fallback_cache[cache_key]
                deleted_count += 1
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Cache delete operation failed: {e}")
            return False
    
    def clear_layer(self, layer: str) -> int:
        """Clear all entries for a specific cache layer."""
        try:
            deleted_count = 0
            prefix = self.key_prefixes.get(layer, 'nlq:general:')
            
            if self.use_redis and self.redis_client:
                # Get all keys with prefix
                keys = self.redis_client.keys(f"{prefix}*")
                if keys:
                    deleted_count += self.redis_client.delete(*keys)
            
            # Clear from fallback cache
            fallback_keys = [k for k in self.fallback_cache.keys() if k.startswith(prefix)]
            for key in fallback_keys:
                del self.fallback_cache[key]
                deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} entries from {layer} cache layer")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache layer clear failed for {layer}: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cache entries."""
        try:
            if self.use_redis and self.redis_client:
                # Clear all NLQ keys
                keys = self.redis_client.keys("nlq:*")
                if keys:
                    self.redis_client.delete(*keys)
            
            # Clear fallback cache
            self.fallback_cache.clear()
            
            # Reset metrics
            self.metrics = CacheMetrics()
            
            logger.info("All cache entries cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear all failed: {e}")
            return False
    
    def _cleanup_fallback_cache(self):
        """Clean up fallback cache when it gets too large."""
        if len(self.fallback_cache) <= self.max_cache_size:
            return
        
        # Sort by expiration time and remove oldest entries
        sorted_entries = sorted(
            self.fallback_cache.items(),
            key=lambda x: x[1]['expires_at']
        )
        
        # Keep only the most recent entries
        entries_to_keep = sorted_entries[-int(self.max_cache_size * 0.8):]
        self.fallback_cache = dict(entries_to_keep)
        
        logger.info(f"Cleaned up fallback cache, kept {len(self.fallback_cache)} entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'metrics': asdict(self.metrics),
            'redis_connected': self.use_redis and self.redis_client is not None,
            'fallback_cache_size': len(self.fallback_cache),
            'configuration': {
                'default_ttl_hours': self.default_ttl / 3600,
                'max_cache_size': self.max_cache_size,
                'compression_enabled': self.compression_enabled
            }
        }
        
        if self.use_redis and self.redis_client:
            try:
                redis_info = self.redis_client.info('memory')
                stats['redis_memory'] = {
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B'),
                    'maxmemory': redis_info.get('maxmemory', 0)
                }
            except:
                pass
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache systems."""
        health = {
            'redis_healthy': False,
            'fallback_healthy': True,
            'overall_status': 'degraded'
        }
        
        # Test Redis connection
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.ping()
                health['redis_healthy'] = True
            except:
                logger.warning("Redis health check failed")
        
        # Test fallback cache
        try:
            test_key = 'health_check_test'
            self.fallback_cache[test_key] = {'test': True}
            del self.fallback_cache[test_key]
        except:
            health['fallback_healthy'] = False
            logger.error("Fallback cache health check failed")
        
        # Determine overall status
        if health['redis_healthy'] or health['fallback_healthy']:
            health['overall_status'] = 'healthy'
        else:
            health['overall_status'] = 'critical'
        
        return health

# Cache layers for different types of data
class CacheLayer:
    """Cache layer management for different data types."""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
    
    def cache_sql_result(self, query_hash: str, result: Any, execution_time: float = 0) -> bool:
        """Cache SQL query result."""
        cache_data = {
            'result': result,
            'execution_time': execution_time,
            'cached_at': datetime.now().isoformat()
        }
        return self.cache.set('sql_results', query_hash, cache_data)
    
    def get_sql_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached SQL result."""
        return self.cache.get('sql_results', query_hash)
    
    def cache_llm_response(self, prompt_hash: str, response: str, model: str, 
                          tokens_used: int = 0) -> bool:
        """Cache LLM response."""
        cache_data = {
            'response': response,
            'model': model,
            'tokens_used': tokens_used,
            'cached_at': datetime.now().isoformat()
        }
        return self.cache.set('llm_responses', prompt_hash, cache_data)
    
    def get_llm_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM response."""
        return self.cache.get('llm_responses', prompt_hash)
    
    def cache_final_response(self, query_hash: str, response_data: Dict[str, Any]) -> bool:
        """Cache complete query response."""
        return self.cache.set('final_responses', query_hash, response_data, ttl=3600)  # 1 hour TTL
    
    def get_final_response(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached final response."""
        return self.cache.get('final_responses', query_hash)
    
    def cache_rag_context(self, context_hash: str, context_data: str) -> bool:
        """Cache RAG context."""
        return self.cache.set('rag_contexts', context_hash, context_data)
    
    def get_rag_context(self, context_hash: str) -> Optional[str]:
        """Get cached RAG context."""
        return self.cache.get('rag_contexts', context_hash)

# Global cache instance
_global_cache = None

def get_cache_system(config: Dict[str, Any] = None) -> Tuple[RedisCache, CacheLayer]:
    """Get global cache system instance."""
    global _global_cache
    
    if _global_cache is None:
        redis_cache = RedisCache(config)
        cache_layer = CacheLayer(redis_cache)
        _global_cache = (redis_cache, cache_layer)
    
    return _global_cache

def reset_cache_system():
    """Reset global cache system."""
    global _global_cache
    if _global_cache:
        _global_cache[0].clear_all()
    _global_cache = None
