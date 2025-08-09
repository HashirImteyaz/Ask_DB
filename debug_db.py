import sqlite3
import os

db_path = 'DATA/plm_updated.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if it's a valid SQLite database
    cursor.execute("SELECT sqlite_version()")
    version = cursor.fetchone()
    print(f"SQLite version: {version[0]}")
    
    # Get all objects in the database (not just tables)
    cursor.execute("SELECT type, name FROM sqlite_master")
    all_objects = cursor.fetchall()
    
    print(f"\nAll objects in database ({len(all_objects)}):")
    for obj_type, name in all_objects:
        print(f"  {obj_type}: {name}")
    
    # Check the schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
    schemas = cursor.fetchall()
    
    if schemas:
        print(f"\nSchemas found ({len(schemas)}):")
        for schema in schemas[:3]:  # Show first 3 schemas
            print(f"  {schema[0][:200]}..." if len(schema[0]) > 200 else f"  {schema[0]}")
    
    # Try to see if there are any hidden or system tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    user_tables = cursor.fetchall()
    print(f"\nUser tables: {user_tables}")
    
    # Check for SQLite internal tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sqlite_%'")
    system_tables = cursor.fetchall()
    print(f"System tables: {system_tables}")
    
    # Check database integrity
    cursor.execute("PRAGMA integrity_check")
    integrity = cursor.fetchone()
    print(f"\nDatabase integrity: {integrity[0]}")
    
    # Check if database is encrypted or has a different format
    cursor.execute("PRAGMA database_list")
    db_info = cursor.fetchall()
    print(f"\nDatabase info: {db_info}")
    
    conn.close()
    
except sqlite3.DatabaseError as e:
    print(f"SQLite database error: {e}")
    print("The file might be corrupted or not a valid SQLite database")
    
    # Check the first few bytes of the file
    try:
        with open(db_path, 'rb') as f:
            header = f.read(16)
            print(f"File header (first 16 bytes): {header}")
            print(f"Header as string: {header.decode('utf-8', errors='ignore')}")
    except Exception as e2:
        print(f"Error reading file header: {e2}")
        
except Exception as e:
    print(f"Unexpected error: {e}")
