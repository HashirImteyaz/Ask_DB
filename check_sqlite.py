import sqlite3
import os

# Check if file exists and get size
db_path = 'DATA/plm_updated.db'
if os.path.exists(db_path):
    size = os.path.getsize(db_path)
    print(f"Database file exists: {db_path}")
    print(f"File size: {size:,} bytes ({size/1024/1024:.1f} MB)")
else:
    print(f"Database file not found: {db_path}")
    exit()

try:
    # Connect directly to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"\nFound {len(tables)} tables:")
    
    total_rows = 0
    for table_tuple in tables:
        table_name = table_tuple[0]
        print(f"\n=== Table: {table_name} ===")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"Columns ({len(columns)}):")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            count = cursor.fetchone()[0]
            total_rows += count
            print(f"Rows: {count:,}")
            
            # Get sample data (first 2 rows)
            if count > 0:
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 2")
                sample_data = cursor.fetchall()
                if sample_data:
                    print("Sample data:")
                    for i, row in enumerate(sample_data, 1):
                        # Truncate long values for display
                        truncated_row = []
                        for val in row:
                            if isinstance(val, str) and len(str(val)) > 50:
                                truncated_row.append(str(val)[:50] + "...")
                            else:
                                truncated_row.append(val)
                        print(f"  Row {i}: {truncated_row}")
                        
        except Exception as e:
            print(f"Error reading table {table_name}: {e}")
    
    print(f"\nTotal rows across all tables: {total_rows:,}")
    
    conn.close()
    
except Exception as e:
    print(f"Error connecting to SQLite database: {e}")
