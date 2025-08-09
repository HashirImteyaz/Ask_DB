import sqlite3
import os

# Check if SQLite database exists and has data
db_path = 'DATA/plm_updated.db'
if os.path.exists(db_path):
    print(f"✅ Database found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if tables:
        print(f"📊 Found {len(tables)} tables:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} rows")
            
            # Show sample data for first few rows
            if count > 0:
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 2;")
                sample_data = cursor.fetchall()
                cursor.execute(f"PRAGMA table_info(`{table_name}`);")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"    Columns: {columns}")
                print(f"    Sample data: {sample_data[:1]}")
            print()
    else:
        print("❌ No tables found in database")
    
    conn.close()
else:
    print(f"❌ Database not found: {db_path}")
