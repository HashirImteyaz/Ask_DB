from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
from datetime import datetime
import os
import json

class ConversationVectorMemory:
    def __init__(self, max_turns=15, config_path=None):
        # Load configuration
        config = {}
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        memory_config = config.get('memory', {})
        self.max_turns = memory_config.get('max_turns', max_turns)
        self.default_recent_history = memory_config.get('default_recent_history', 15)
        self.optional_retrieval_count = memory_config.get('optional_retrieval_count', 5)
        self.optional_recent_count = memory_config.get('optional_recent_count', 10)
        self.similarity_threshold = memory_config.get('similarity_threshold', 0.5)
        self.response_truncation_limit = memory_config.get('response_truncation_limit', 500)
            
        self.history: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        
        # Initialize OpenAI client with error handling
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client in memory module: {e}")
            # Create a mock client that will handle the error gracefully
            self.client = None

    def add(self, user_query: str, agent_response: str, sql: Optional[str] = None):
        entry = {
            "query": user_query,
            "response": agent_response[:self.response_truncation_limit],  # Use config value
            "sql": sql if sql is not None else "",
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate embedding for the query using correct API
        try:
            if self.client is None:
                # Fallback: add to history without embedding
                print("Warning: OpenAI client not available, adding to history without embedding")
                self.history.append(entry)
                # Add a zero vector as placeholder
                self.embeddings.append(np.zeros(1536))  # text-embedding-3-small dimension
            else:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=user_query
                )
                embedding = np.array(response.data[0].embedding)
                
                self.history.append(entry)
                self.embeddings.append(embedding)
            
            # Maintain max_turns limit
            if len(self.history) > self.max_turns:
                self.history = self.history[-self.max_turns:]
                self.embeddings = self.embeddings[-self.max_turns:]
                
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Still add to history without embedding for graceful degradation
            self.history.append(entry)
            if len(self.history) > self.max_turns:
                self.history = self.history[-self.max_turns:]

    def search(self, current_query: str, top_k=3, use_retrieval=False) -> List[Dict]:
        """
        Search for relevant conversation history.
        
        Args:
            current_query: The current user query
            top_k: Number of results to return (used when use_retrieval=False)
            use_retrieval: If True, combines retrieval + recent. If False, returns recent only.
        
        Returns:
            List of relevant conversation entries
        """
        if not self.history:
            return []
        
        if not use_retrieval:
            # Default behavior: return last 15 conversations (or configured amount)
            return self.get_recent_context(self.default_recent_history)
        
        # Optional retrieval mode: combine retrieval + recent history
        retrieval_results = []
        recent_results = []
        
        # Get retrieval results if embeddings are available
        if self.embeddings and self.client is not None:
            try:
                # Generate embedding for current query using correct API
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=current_query
                )
                query_embedding = np.array(response.data[0].embedding)
                
                # Calculate similarities
                similarities = []
                for i, emb in enumerate(self.embeddings):
                    # Skip zero vectors (placeholders)
                    if np.any(emb):
                        similarity = np.dot(query_embedding, emb) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                        )
                        similarities.append((i, similarity))
                
                # Sort by similarity and get top results above threshold
                similarities.sort(key=lambda x: x[1], reverse=True)
                retrieval_results = [
                    self.history[i] for i, score in similarities[:self.optional_retrieval_count] 
                    if score > self.similarity_threshold
                ]
                
            except Exception as e:
                print(f"Error in similarity search: {e}")
        
        # Get recent results (excluding those already in retrieval_results)
        recent_results = self.get_recent_context(self.optional_recent_count)
        
        # Combine and deduplicate results
        combined_results = []
        seen_timestamps = set()
        
        # Add retrieval results first
        for result in retrieval_results:
            if result['timestamp'] not in seen_timestamps:
                combined_results.append(result)
                seen_timestamps.add(result['timestamp'])
        
        # Add recent results
        for result in recent_results:
            if result['timestamp'] not in seen_timestamps:
                combined_results.append(result)
                seen_timestamps.add(result['timestamp'])
        
        return combined_results

    def get_recent_context(self, last_n=None) -> List[Dict]:
        """Get the most recent conversation entries."""
        if last_n is None:
            last_n = self.default_recent_history
        return self.history[-last_n:] if self.history else []
