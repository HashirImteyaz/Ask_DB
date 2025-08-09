import sqlite3
import os
from sqlalchemy import create_engine, text

# Check SQLite
print("=== Checking SQLite Database ===")
if os.path.exists('DATA/plm_updated.db'):
    conn = sqlite3.connect('DATA/plm_updated.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print('SQLite tables:', [t[0] for t in tables])
    
    if tables:
        # Check the first table
        first_table = tables[0][0]
        cursor.execute(f'SELECT * FROM "{first_table}" LIMIT 3')
        rows = cursor.fetchall()
        print(f'\nSample data from {first_table}:')
        for row in rows:
            print(row)
    conn.close()
else:
    print('SQLite database not found')

# Check PostgreSQL
print("\n=== Checking PostgreSQL Database ===")
try:
    engine = create_engine('postgresql://nlq_user:sa@localhost:5432/nlq_plm')
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in result.fetchall()]
        print('PostgreSQL tables:', tables)
        
        if tables:
            # Check the first table
            first_table = tables[0]
            result = conn.execute(text(f'SELECT * FROM "{first_table}" LIMIT 3'))
            rows = result.fetchall()
            print(f'\nSample data from {first_table}:')
            for row in rows:
                print(row)
except Exception as e:
    print('PostgreSQL error:', str(e))

print("\n=== Environment Variables ===")
print('DATABASE_URL:', os.getenv('DATABASE_URL'))
