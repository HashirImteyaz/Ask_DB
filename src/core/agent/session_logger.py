# session_logger.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any
import tiktoken
from pathlib import Path
import re
import html

class SessionLogger:
    """
    Comprehensive session logger that tracks:
    - User queries and responses
    - SQL queries generated
    - LLM token usage
    - Session metadata
    """
    
    def __init__(self, logs_dir: str = "logs", config: Dict = None):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Load config or use defaults
        self.config = config or {}
        log_config = self.config.get('logging', {})
        
        # Session tracking
        self.session_id = self._generate_session_id()
        self.log_file_path = self.logs_dir / f"log_{self.session_id}.log"
        self.total_tokens = 0
        self.total_queries = 0
        
        # Token encoder for counting
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            print(f"Warning: Could not initialize tiktoken encoder for gpt-4: {e}")
            try:
                # Fallback encoder
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e2:
                print(f"Warning: Could not initialize fallback tiktoken encoder: {e2}")
                # Use a dummy encoder that just estimates tokens
                self.encoder = None
        
        # Initialize log file
        self._initialize_log_file()
    
    def _generate_session_id(self) -> str:
        """Generate session ID with date and time"""
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to human-readable plain text"""
        if not html_content or not isinstance(html_content, str):
            return str(html_content) if html_content else ""
        
        # Unescape HTML entities first
        text = html.unescape(html_content)
        
        # Replace table structures with readable format
        text = re.sub(r'<table[^>]*>', '\n[TABLE]\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</table>', '\n[/TABLE]\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<thead[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</thead>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<tbody[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</tbody>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<tr[^>]*>', '\n| ', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>', ' |', text, flags=re.IGNORECASE)
        text = re.sub(r'<th[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</th>', ' | ', text, flags=re.IGNORECASE)
        text = re.sub(r'<td[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</td>', ' | ', text, flags=re.IGNORECASE)
        
        # Replace common HTML elements with readable equivalents
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<div[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<h[1-6][^>]*>', '\n## ', text, flags=re.IGNORECASE)
        text = re.sub(r'</h[1-6]>', ' ##\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<strong[^>]*>|<b[^>]*>', '**', text, flags=re.IGNORECASE)
        text = re.sub(r'</strong>|</b>', '**', text, flags=re.IGNORECASE)
        text = re.sub(r'<em[^>]*>|<i[^>]*>', '*', text, flags=re.IGNORECASE)
        text = re.sub(r'</em>|</i>', '*', text, flags=re.IGNORECASE)
        
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def _initialize_log_file(self):
        """Initialize the log file with session header"""
        header = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "system": "NLQ PLM System",
            "version": "1.0"
        }
        
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NLQ PLM SYSTEM - SESSION LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write(f"Log File: {self.log_file_path.name}\n")
            f.write("="*80 + "\n\n")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if not text:
            return 0
        if self.encoder is None:
            # Use rough estimation when encoder is not available
            return int(len(str(text).split()) * 1.3)
        try:
            return len(self.encoder.encode(str(text)))
        except Exception as e:
            # Fallback: rough estimation
            return int(len(str(text).split()) * 1.3)
    
    def log_query_session(self, 
                         user_query: str,
                         sql_query: Optional[str] = None,
                         agent_response: str = "",
                         llm_calls: list = None,
                         context_used: Optional[str] = None,
                         error: Optional[str] = None,
                         metadata: Optional[Dict] = None):
        """
        Log a complete query session with all details
        
        Args:
            user_query: Original user question
            sql_query: Generated SQL query
            agent_response: Final agent response
            llm_calls: List of LLM calls made (with prompts and responses)
            context_used: Context retrieved from memory/RAG
            error: Any error that occurred
            metadata: Additional metadata
        """
        
        self.total_queries += 1
        timestamp = datetime.now().isoformat()
        
        # Calculate token usage
        query_tokens = self.count_tokens(user_query)
        response_tokens = self.count_tokens(agent_response)
        sql_tokens = self.count_tokens(sql_query) if sql_query else 0
        context_tokens = self.count_tokens(context_used) if context_used else 0
        
        # Calculate LLM call tokens
        llm_tokens = 0
        if llm_calls:
            for call in llm_calls:
                llm_tokens += self.count_tokens(call.get('prompt', ''))
                llm_tokens += self.count_tokens(call.get('response', ''))
        
        total_session_tokens = query_tokens + response_tokens + sql_tokens + context_tokens + llm_tokens
        self.total_tokens += total_session_tokens
        
        # Create log entry
        log_entry = {
            "query_id": self.total_queries,
            "timestamp": timestamp,
            "user_query": user_query,
            "sql_query": sql_query,
            "agent_response": agent_response,
            "token_usage": {
                "query_tokens": query_tokens,
                "response_tokens": response_tokens,
                "sql_tokens": sql_tokens,
                "context_tokens": context_tokens,
                "llm_tokens": llm_tokens,
                "total_tokens": total_session_tokens,
                "session_total": self.total_tokens
            },
            "llm_calls": llm_calls or [],
            "context_used": context_used,
            "error": error,
            "metadata": metadata or {}
        }
        
        # Write to log file
        self._write_log_entry(log_entry)
        
        return log_entry
    
    def _write_log_entry(self, entry: Dict):
        """Write a formatted log entry to the file"""
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"QUERY #{entry['query_id']} - {entry['timestamp']}\n")
            f.write(f"{'='*60}\n")
            
            # User Query
            f.write(f"\n🔤 USER QUERY:\n")
            f.write(f"{entry['user_query']}\n")
            
            # SQL Query
            if entry['sql_query']:
                f.write(f"\n🔍 GENERATED SQL:\n")
                f.write(f"{entry['sql_query']}\n")
            
            # Agent Response
            f.write(f"\n🤖 AGENT RESPONSE:\n")
            # Convert HTML to human-readable text for logging
            readable_response = self._html_to_text(entry['agent_response'])
            f.write(f"{readable_response}\n")
            
            # Token Usage
            tokens = entry['token_usage']
            f.write(f"\n📊 TOKEN USAGE:\n")
            f.write(f"  Query Tokens: {tokens['query_tokens']}\n")
            f.write(f"  Response Tokens: {tokens['response_tokens']}\n")
            f.write(f"  SQL Tokens: {tokens['sql_tokens']}\n")
            f.write(f"  Context Tokens: {tokens['context_tokens']}\n")
            f.write(f"  LLM Call Tokens: {tokens['llm_tokens']}\n")
            f.write(f"  Total This Query: {tokens['total_tokens']}\n")
            f.write(f"  Session Total: {tokens['session_total']}\n")
            
            # LLM Calls Detail
            if entry['llm_calls']:
                f.write(f"\n🧠 LLM CALLS ({len(entry['llm_calls'])}):\n")
                for i, call in enumerate(entry['llm_calls'], 1):
                    f.write(f"  Call #{i}: {call.get('type', 'Unknown')}\n")
                    f.write(f"    Model: {call.get('model', 'Unknown')}\n")
                    f.write(f"    Tokens: {self.count_tokens(call.get('prompt', ''))} + {self.count_tokens(call.get('response', ''))}\n")
            
            # Context Used
            if entry['context_used']:
                f.write(f"\n📝 CONTEXT USED:\n")
                f.write(f"{entry['context_used'][:500]}{'...' if len(entry['context_used']) > 500 else ''}\n")
            
            # Error
            if entry['error']:
                f.write(f"\n❌ ERROR:\n")
                f.write(f"{entry['error']}\n")
            
            # Metadata
            if entry['metadata']:
                f.write(f"\n📋 METADATA:\n")
                f.write(f"{json.dumps(entry['metadata'], indent=2)}\n")
            
            f.write(f"\n{'='*60}\n")
    
    def log_llm_call(self, call_type: str, model: str, prompt: str, response: str, metadata: Dict = None):
        """Log individual LLM call details"""
        return {
            "type": call_type,
            "model": model,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": self.count_tokens(prompt),
            "response_tokens": self.count_tokens(response),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
    
    def finalize_session(self):
        """Write session summary at the end"""
        end_time = datetime.now()
        
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write("SESSION SUMMARY\n")
            f.write(f"{'='*80}\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"End Time: {end_time.isoformat()}\n")
            f.write(f"Total Queries: {self.total_queries}\n")
            f.write(f"Total Tokens: {self.total_tokens}\n")
            f.write(f"Average Tokens per Query: {self.total_tokens / max(1, self.total_queries):.2f}\n")
            f.write(f"Log File: {self.log_file_path.name}\n")
            f.write(f"{'='*80}\n")
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        return {
            "session_id": self.session_id,
            "total_queries": self.total_queries,
            "total_tokens": self.total_tokens,
            "average_tokens_per_query": self.total_tokens / max(1, self.total_queries),
            "log_file": str(self.log_file_path)
        }


# Global session logger instance
_session_logger = None

def get_session_logger(config: Dict = None) -> SessionLogger:
    """Get or create the global session logger"""
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger(config=config)
    return _session_logger

def reset_session_logger(config: Dict = None) -> SessionLogger:
    """Reset the session logger (for new sessions)"""
    global _session_logger
    if _session_logger:
        _session_logger.finalize_session()
    _session_logger = SessionLogger(config=config)
    return _session_logger