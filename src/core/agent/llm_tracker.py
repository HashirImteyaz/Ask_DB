# llm_tracker.py

from typing import Dict, List, Optional, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import tiktoken
from datetime import datetime
import os
import openai
from openai import OpenAI

class TokenTrackingLLM:
    """
    Direct OpenAI wrapper that tracks token usage for each call
    Avoids Pydantic issues with ChatOpenAI
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0, **kwargs):
        # Initialize OpenAI client directly
        api_key = kwargs.get('openai_api_key') or kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.call_history: List[Dict] = []
        
        # Initialize token encoder
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except Exception:
            # Fallback encoder
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if not text:
            return 0
        try:
            return len(self.encoder.encode(str(text)))
        except Exception:
            # Fallback: rough estimation
            return int(len(str(text).split()) * 1.3)
    
    def invoke(self, prompt: Union[str, List[Dict]], call_type: str = "general") -> AIMessage:
        """
        Invoke the LLM and track token usage
        
        Args:
            prompt: The input prompt (string or messages list)
            call_type: Type of call (e.g., "sql_generation", "clarification", "final_answer")
        
        Returns:
            AIMessage response from LLM
        """
        start_time = datetime.now()
        
        # Convert prompt to messages format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            prompt_text = prompt
        elif isinstance(prompt, list):
            messages = prompt
            prompt_text = "\n".join([msg.get("content", "") for msg in messages])
        else:
            # Handle BaseMessage objects
            prompt_text = str(prompt)
            messages = [{"role": "user", "content": prompt_text}]
        
        # Count input tokens
        input_tokens = self.count_tokens(prompt_text)
        
        try:
            # Make the OpenAI API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Count output tokens
            output_tokens = self.count_tokens(response_content)
            
            # Get actual token usage from API if available
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Record the call
            call_record = {
                "timestamp": start_time.isoformat(),
                "call_type": call_type,
                "model": self.model,
                "prompt": prompt_text,
                "response": response_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "duration_seconds": duration
            }
            
            self.call_history.append(call_record)
            
            # Return AIMessage to match LangChain interface
            return AIMessage(content=response_content)
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return error as AIMessage
            error_msg = f"Error: {str(e)}"
            return AIMessage(content=error_msg)
    
    def get_call_history(self) -> List[Dict]:
        """Get the history of all LLM calls"""
        return self.call_history.copy()
    
    def get_total_tokens(self) -> int:
        """Get total tokens used across all calls"""
        return sum(call["total_tokens"] for call in self.call_history)
    
    def get_call_stats(self) -> Dict:
        """Get statistics about LLM calls"""
        if not self.call_history:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "average_tokens_per_call": 0,
                "total_duration": 0
            }
        
        total_calls = len(self.call_history)
        total_tokens = sum(call["total_tokens"] for call in self.call_history)
        total_input_tokens = sum(call["input_tokens"] for call in self.call_history)
        total_output_tokens = sum(call["output_tokens"] for call in self.call_history)
        total_duration = sum(call["duration_seconds"] for call in self.call_history)
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "average_tokens_per_call": total_tokens / total_calls,
            "total_duration": total_duration,
            "calls_by_type": self._group_calls_by_type()
        }
    
    def _group_calls_by_type(self) -> Dict:
        """Group calls by type with statistics"""
        grouped = {}
        for call in self.call_history:
            call_type = call["call_type"]
            if call_type not in grouped:
                grouped[call_type] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_duration": 0
                }
            
            grouped[call_type]["count"] += 1
            grouped[call_type]["total_tokens"] += call["total_tokens"]
            grouped[call_type]["total_input_tokens"] += call["input_tokens"]
            grouped[call_type]["total_output_tokens"] += call["output_tokens"]
            grouped[call_type]["total_duration"] += call["duration_seconds"]
        
        # Calculate averages
        for call_type in grouped:
            count = grouped[call_type]["count"]
            grouped[call_type]["average_tokens"] = grouped[call_type]["total_tokens"] / count
            grouped[call_type]["average_duration"] = grouped[call_type]["total_duration"] / count
        
        return grouped
    
    def reset_history(self):
        """Reset the call history"""
        self.call_history = []


class LLMCallTracker:
    """
    Global tracker for all LLM calls in a session
    """
    
    def __init__(self):
        self.all_calls: List[Dict] = []
        self.llm_instances: Dict[str, TokenTrackingLLM] = {}
    
    def register_llm(self, name: str, llm: TokenTrackingLLM):
        """Register an LLM instance for tracking"""
        self.llm_instances[name] = llm
    
    def get_all_calls(self) -> List[Dict]:
        """Get all calls from all registered LLMs"""
        all_calls = []
        for name, llm in self.llm_instances.items():
            for call in llm.get_call_history():
                call_copy = call.copy()
                call_copy["llm_name"] = name
                all_calls.append(call_copy)
        
        # Sort by timestamp
        all_calls.sort(key=lambda x: x["timestamp"])
        return all_calls
    
    def get_session_stats(self) -> Dict:
        """Get comprehensive session statistics"""
        all_calls = self.get_all_calls()
        
        if not all_calls:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "llms_used": [],
                "calls_by_llm": {},
                "calls_by_type": {}
            }
        
        total_tokens = sum(call["total_tokens"] for call in all_calls)
        total_input_tokens = sum(call["input_tokens"] for call in all_calls)
        total_output_tokens = sum(call["output_tokens"] for call in all_calls)
        
        # Group by LLM
        calls_by_llm = {}
        for call in all_calls:
            llm_name = call["llm_name"]
            if llm_name not in calls_by_llm:
                calls_by_llm[llm_name] = {"count": 0, "tokens": 0}
            calls_by_llm[llm_name]["count"] += 1
            calls_by_llm[llm_name]["tokens"] += call["total_tokens"]
        
        # Group by type
        calls_by_type = {}
        for call in all_calls:
            call_type = call["call_type"]
            if call_type not in calls_by_type:
                calls_by_type[call_type] = {"count": 0, "tokens": 0}
            calls_by_type[call_type]["count"] += 1
            calls_by_type[call_type]["tokens"] += call["total_tokens"]
        
        return {
            "total_calls": len(all_calls),
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "llms_used": list(self.llm_instances.keys()),
            "calls_by_llm": calls_by_llm,
            "calls_by_type": calls_by_type
        }
    
    def reset_all(self):
        """Reset all tracking data"""
        for llm in self.llm_instances.values():
            llm.reset_history()
        self.all_calls = []


# Global tracker instance
_global_tracker = LLMCallTracker()

def get_global_tracker() -> LLMCallTracker:
    """Get the global LLM call tracker"""
    return _global_tracker

def reset_global_tracker():
    """Reset the global tracker"""
    global _global_tracker
    _global_tracker.reset_all()