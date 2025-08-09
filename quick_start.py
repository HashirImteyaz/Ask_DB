#!/usr/bin/env python3
"""
Quick Start Script for NLQ PLM System
Automates the setup and initialization of the enhanced system
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_banner():
    """Print the system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🚀 Enhanced NLQ PLM System - Quick Start 🚀          ║
    ║                                                              ║
    ║        Truly Powerful • Intelligent • Robust                ║
    ║        Handle Any Query • Any Database • Any Scale          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required. Current version:", f"{version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor} detected")

def check_redis_connection():
    """Check if Redis is available"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    # Check if enhanced requirements exist, otherwise use standard
    req_file = "requirements_enhanced.txt"
    if not Path(req_file).exists():
        req_file = "requirements.txt"
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_environment():
    """Set up environment variables"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create basic .env file
    env_content = """# NLQ PLM System Configuration

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///DATA/plm_updated.db

# Redis Configuration
REDIS_URL=redis://localhost:6379/1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Feature Flags
ASYNC_PROCESSING_ENABLED=true
CACHING_ENABLED=true
SQL_VALIDATION_ENABLED=true
DATABASE_STATISTICS_ENABLED=true

# Logging
LOG_LEVEL=INFO
"""
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print("✅ .env file created (please update OPENAI_API_KEY)")
    return True

def check_openai_key():
    """Check if OpenAI API key is set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️  OpenAI API key not set. Please update .env file with your API key")
        return False
    
    print("✅ OpenAI API key configured")
    return True

def setup_database():
    """Set up database directory"""
    data_dir = Path("DATA")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("✅ DATA directory created")
    else:
        print("✅ DATA directory exists")
    return True

def start_redis_instructions():
    """Provide instructions for starting Redis"""
    system = platform.system().lower()
    
    print("\n🔴 Redis Setup Instructions:")
    
    if system == "windows":
        print("Windows:")
        print("1. Install Redis using Chocolatey: choco install redis-64")
        print("2. Start Redis: redis-server")
        print("3. Or download Redis for Windows from: https://github.com/microsoftarchive/redis/releases")
    elif system == "darwin":
        print("macOS:")
        print("1. Install Redis: brew install redis")
        print("2. Start Redis: brew services start redis")
    else:
        print("Linux:")
        print("1. Install Redis: sudo apt install redis-server")
        print("2. Start Redis: sudo systemctl start redis-server")

def run_system_check():
    """Run comprehensive system check"""
    print("\n🔍 Running system checks...")
    
    checks = {
        "Python Version": check_python_version,
        "Database Setup": setup_database,
        "Environment Setup": setup_environment,
        "Requirements Install": install_requirements,
    }
    
    results = {}
    for check_name, check_func in checks.items():
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results[check_name] = False
    
    # Check optional components
    redis_available = check_redis_connection()
    openai_configured = False
    
    try:
        openai_configured = check_openai_key()
    except:
        pass
    
    return all(results.values()), redis_available, openai_configured

def start_application():
    """Start the application"""
    print("\n🚀 Starting NLQ PLM System...")
    
    try:
        # Import and start the application
        os.environ["PYTHONPATH"] = str(Path.cwd())
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.api.main_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
        time.sleep(3)  # Give server time to start
        
        print("✅ System started successfully!")
        print("\n📍 Access Points:")
        print("   • Web Interface: http://localhost:8000/")
        print("   • API Documentation: http://localhost:8000/docs")
        print("   • Health Check: http://localhost:8000/health")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n📋 Next Steps:")
    print("1. Update .env file with your OpenAI API key")
    print("2. Start Redis server (see instructions above)")
    print("3. Upload your data files through the web interface")
    print("4. Start querying your data with natural language!")
    print("\n💡 Pro Tips:")
    print("• Use the web interface for easy file uploads")
    print("• Try complex queries like 'Show top 10 products by revenue'")
    print("• Monitor system health at /health endpoint")
    print("• Check cache performance at /system/cache-stats")

def main():
    """Main setup function"""
    print_banner()
    
    # Run system checks
    core_ready, redis_ready, openai_ready = run_system_check()
    
    if not core_ready:
        print("\n❌ Core system setup failed. Please check errors above.")
        sys.exit(1)
    
    print("\n🎉 Core system setup completed!")
    
    # Show status summary
    print("\n📊 System Status:")
    print(f"   • Core System: ✅ Ready")
    print(f"   • Redis: {'✅ Ready' if redis_ready else '⚠️  Not Available'}")
    print(f"   • OpenAI API: {'✅ Configured' if openai_ready else '⚠️  Needs Configuration'}")
    
    if not redis_ready:
        start_redis_instructions()
    
    if not openai_ready:
        print("\n⚠️  Please update your OpenAI API key in .env file before starting queries")
    
    # Ask if user wants to start the application
    if redis_ready and openai_ready:
        response = input("\n🚀 Start the application now? (y/N): ")
        if response.lower() in ['y', 'yes']:
            start_application()
    
    show_next_steps()
    
    print("\n🌟 Welcome to the Enhanced NLQ PLM System!")
    print("For detailed setup instructions, see DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main()
