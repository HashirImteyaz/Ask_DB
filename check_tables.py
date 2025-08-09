import os
from sqlalchemy import create_engine, inspect, text

# Get the database URL from environment
DB_URL = os.getenv("DATABASE_URL", "sqlite:///DATA/plm_updated.db")
print(f"Connecting to database: {DB_URL}")

try:
    # Connect to the database
    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    
    # Get all table names
    tables = inspector.get_table_names()
    print(f"\nFound {len(tables)} tables in the database:")
    
    if not tables:
        print("❌ No tables found! This explains why queries are failing.")
        print("You need to update DATABASE_URL to point to your actual database.")
    else:
        print("✅ Tables found:")
        for table_name in tables:
            print(f"  - {table_name}")
            
            # Get column info
            columns = inspector.get_columns(table_name)
            print(f"    Columns ({len(columns)}):")
            for col in columns[:5]:  # Show first 5 columns
                print(f"      - {col['name']} ({col['type']})")
            if len(columns) > 5:
                print(f"      ... and {len(columns) - 5} more columns")
            
            # Get row count
            with engine.connect() as conn:
                try:
                    result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                    count = result.scalar()
                    print(f"    Rows: {count:,}")
                except Exception as e:
                    print(f"    Rows: Error getting count - {e}")
            print()

except Exception as e:
    print(f"❌ Failed to connect to database: {e}")
    print("Please check your DATABASE_URL in the .env file")
