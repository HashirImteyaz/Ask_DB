#!/usr/bin/env python3
"""
Installation validator for NLQ PLM System
This script checks if all required dependencies are properly installed.
"""
import sys
import importlib
import pkg_resources
from typing import Dict, List, Tuple

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and optionally verify version."""
    try:
        pkg = importlib.import_module(package_name)
        if hasattr(pkg, '__version__'):
            version = pkg.__version__
        else:
            try:
                version = pkg_resources.get_distribution(package_name.split('.')[0]).__version__
            except:
                version = "unknown"
        
        if min_version:
            try:
                installed_version = pkg_resources.parse_version(version)
                required_version = pkg_resources.parse_version(min_version)
                if installed_version >= required_version:
                    return True, f"✅ {package_name} {version}"
                else:
                    return False, f"❌ {package_name} {version} (requires >= {min_version})"
            except:
                return True, f"⚠️  {package_name} {version} (version check failed)"
        else:
            return True, f"✅ {package_name} {version}"
    except ImportError:
        return False, f"❌ {package_name} not installed"
    except Exception as e:
        return False, f"❌ {package_name} error: {e}"

def validate_installation():
    """Validate the NLQ PLM system installation."""
    print("🔍 Validating NLQ PLM System Installation...\n")
    
    # Core requirements
    core_packages = [
        ("fastapi", "0.104.0"),
        ("uvicorn", "0.24.0"),
        ("sqlalchemy", "2.0.0"),
        ("pydantic", "2.4.0"),
        ("pandas", "2.0.0"),
        ("numpy", "1.24.0"),
        ("requests", "2.31.0"),
        ("matplotlib", "3.7.0"),
    ]
    
    # AI/ML packages
    ai_packages = [
        ("openai", "1.40.0"),
        ("tiktoken", "0.7.0"),
        ("sklearn", "1.3.0"),
        ("langchain_core", None),
        ("langchain_openai", None),
        ("langgraph", None),
    ]
    
    # Optional packages
    optional_packages = [
        ("redis", None),
        ("sqlparse", None),
        ("celery", None),
        ("streamlit", "1.28.0"),
        ("llama_index", None),
    ]
    
    all_good = True
    
    # Check core packages
    print("📦 Core Packages:")
    for package, min_version in core_packages:
        success, message = check_package(package, min_version)
        print(f"  {message}")
        if not success:
            all_good = False
    
    print("\n🤖 AI/ML Packages:")
    for package, min_version in ai_packages:
        success, message = check_package(package, min_version)
        print(f"  {message}")
        if not success and package in ["openai", "tiktoken", "sklearn"]:
            all_good = False
    
    print("\n🔧 Optional Packages:")
    for package, min_version in optional_packages:
        success, message = check_package(package, min_version)
        print(f"  {message}")
    
    # Test basic functionality
    print("\n🧪 Functional Tests:")
    
    # Test basic imports
    test_imports = [
        "src.core.agent.sql_logic",
        "src.core.agent.agent_state", 
        "src.api.main_app"
    ]
    
    for module in test_imports:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            all_good = False
    
    # Test environment variables
    import os
    print("\n🌍 Environment:")
    if os.getenv("OPENAI_API_KEY"):
        print("  ✅ OPENAI_API_KEY is set")
    else:
        print("  ⚠️  OPENAI_API_KEY not set (required for AI features)")
        
    if os.path.exists("config.json"):
        print("  ✅ config.json found")
    else:
        print("  ⚠️  config.json not found (will use defaults)")
    
    # Final verdict
    print(f"\n{'='*50}")
    if all_good:
        print("🎉 Installation validation PASSED!")
        print("Your NLQ PLM system is ready to use.")
        print("\nTo start the system:")
        print("  API: python -m uvicorn src.api.main_app:app --reload")
        print("  UI:  streamlit run src/ui/streamlit_chat.py")
    else:
        print("❌ Installation validation FAILED!")
        print("Please install missing packages and try again.")
        print("\nFor help, see INSTALLATION_GUIDE.md")
    
    return all_good

if __name__ == "__main__":
    success = validate_installation()
    sys.exit(0 if success else 1)
