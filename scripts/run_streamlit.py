#!/usr/bin/env python3
"""
Streamlit Chat Interface Launcher
"""
import subprocess
import sys
import requests
import time

def check_api_status():
    """Check if the main API is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Starting Streamlit Chat Interface...")
    print("=" * 50)
    
    # Check if API is running
    print("🔍 Checking API status...")
    if check_api_status():
        print("✅ Main API is running on http://127.0.0.1:8000")
    else:
        print("❌ Main API is not accessible on http://127.0.0.1:8000")
        print("Please start the main API first by running:")
        print("   python main_app.py")
        print()
        choice = input("Continue anyway? (y/N): ").lower()
        if choice != 'y':
            return
    
    print()
    print("🌟 Starting Streamlit app...")
    print("📱 The chat interface will open in your browser")
    print("🌐 URL: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_chat.py",
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