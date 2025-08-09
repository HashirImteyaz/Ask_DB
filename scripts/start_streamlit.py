#!/usr/bin/env python3
"""
Script to start the Streamlit chat interface
"""
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_api_status():
    """Check if the main API is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Start the Streamlit interface"""
    print("Starting NLQ PLM Streamlit Interface...")
    print("=" * 45)
    
    # Check if API is running
    print("Checking API status...")
    if check_api_status():
        print("✅ Main API is running on http://127.0.0.1:8000")
    else:
        print("❌ Main API is not accessible on http://127.0.0.1:8000")
        print("Please start the main API first by running:")
        print("   python scripts/start_api.py")
        print()
        choice = input("Continue anyway? (y/N): ").lower()
        if choice != 'y':
            return
    
    print()
    print("Starting Streamlit app...")
    print("The chat interface will open in your browser")
    print("URL: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application")
    print("=" * 45)
    
    try:
        # Get the path to streamlit_chat.py
        streamlit_path = Path(__file__).parent.parent / "src" / "ui" / "streamlit_chat.py"
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(streamlit_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("Make sure streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()