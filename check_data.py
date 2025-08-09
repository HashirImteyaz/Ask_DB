import sqlite3

def check_database_data():
    conn = sqlite3.connect('DATA/plm_updated.db')
    
    # Get counts
    spec_count = conn.execute('SELECT COUNT(*) FROM Specifications').fetchone()[0]
    recipe_count = conn.execute('SELECT COUNT(*) FROM RecipeExplosion').fetchone()[0]
    
    print(f"Database contains:")
    print(f"  - Specifications: {spec_count} records")
    print(f"  - RecipeExplosion: {recipe_count} records")
    
    # Get sample recipes (CUC = Consumer Unit Content = finished recipes)
    print(f"\nSample recipes:")
    recipes = conn.execute(
        "SELECT SpecCode, SpecDescription FROM Specifications WHERE SpecGroupCode = 'CUC' LIMIT 10"
    ).fetchall()
    
    for code, desc in recipes:
        print(f"  {code}: {desc}")
    
    # Get sample ingredients
    print(f"\nSample ingredients:")
    ingredients = conn.execute(
        "SELECT SpecCode, SpecDescription FROM Specifications WHERE SpecGroupCode = 'ING' LIMIT 10"
    ).fetchall()
    
    for code, desc in ingredients:
        print(f"  {code}: {desc}")
    
    conn.close()

if __name__ == "__main__":
    check_database_data()
