import sqlite3

def test_ingredient_query():
    conn = sqlite3.connect('DATA/plm_updated.db')
    
    # Test the correct query for ingredients
    query = "SELECT DISTINCT SpecDescription FROM Specifications WHERE SpecGroupCode = 'ING' LIMIT 10"
    results = conn.execute(query).fetchall()
    
    print('Sample ingredients from your database:')
    for row in results:
        print(f'  {row[0]}')
    
    # Get total count
    count_query = "SELECT COUNT(DISTINCT SpecDescription) FROM Specifications WHERE SpecGroupCode = 'ING'"
    total = conn.execute(count_query).fetchone()[0]
    print(f'\nTotal unique ingredients in database: {total}')
    
    # Also check SpecGroupCode values to confirm data structure
    codes_query = "SELECT DISTINCT SpecGroupCode, COUNT(*) FROM Specifications GROUP BY SpecGroupCode"
    codes = conn.execute(codes_query).fetchall()
    print(f'\nSpecGroupCode distribution:')
    for code, count in codes:
        print(f'  {code}: {count} records')
    
    conn.close()

if __name__ == "__main__":
    test_ingredient_query()
