#!/usr/bin/env python3
"""
Rebuild RAG System for Existing Database
This script rebuilds the RAG system using existing data in the SQLite database.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def rebuild_rag_for_existing_data():
    """Rebuild RAG system using existing SQLite database."""
    
    print("🔄 Rebuilding RAG system for existing database...")
    
    # Check if database exists
    db_path = "DATA/plm_updated.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Check if schema description exists
    schema_path = "src/config/schema_description.json"
    if not os.path.exists(schema_path):
        print(f"❌ Schema description not found: {schema_path}")
        print("Please upload the schema_description.json file through the web interface first.")
        return False
    
    try:
        # Import the task functions
        from src.core.data_processing.tasks import rebuild_rag_system
        
        print(f"✅ Using database: {db_path}")
        print(f"✅ Using schema: {schema_path}")
        
        # Rebuild the RAG system
        db_url = f"sqlite:///{db_path}"
        result = rebuild_rag_system(db_url=db_url, config_path="config.json")
        
        if result.get('success'):
            print("✅ RAG system rebuilt successfully!")
            print(f"📊 Tables processed: {result.get('tables_count', 0)}")
            print(f"⏱️  Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")
            
            # Trigger a schema update as well
            print("🔄 Updating schema cache...")
            try:
                from src.core.data_processing.utils import update_schema_cache
                update_schema_cache(db_url)
                print("✅ Schema cache updated!")
            except Exception as e:
                print(f"⚠️  Schema cache update failed: {e}")
            
            return True
        else:
            print(f"❌ RAG rebuild failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error rebuilding RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting RAG system rebuild...")
    success = rebuild_rag_for_existing_data()
    
    if success:
        print("\n🎉 RAG system is ready! You can now query your data.")
        print("\nNext steps:")
        print("1. Restart the server (Ctrl+C, then run the server again)")
        print("2. Try your query: 'Which countries are involved in making these recipes?'")
    else:
        print("\n❌ RAG system rebuild failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
