#!/usr/bin/env python3
"""
Script to start the NLQ API server
"""
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Start the API server"""
    print("Starting NLQ PLM API Server...")
    print("=" * 40)
    
    try:
        from src.api.main_app import app
        import uvicorn
        
        print("API Server starting on http://127.0.0.1:8000")
        print("Press Ctrl+C to stop the server")
        print("=" * 40)
        
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        print("And that all dependencies are installed: pip install -r src/config/requirements.txt")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()