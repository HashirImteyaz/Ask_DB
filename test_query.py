import sqlite3
import pandas as pd

# Test the failing SQL query
conn = sqlite3.connect('DATA/plm_updated.db')

query1 = "SELECT DISTINCT r.CUCRecipeCode, r.CUCPlantDescription FROM RecipeExplosion r WHERE UPPER(r.CUCPlantOwnershipFlag) = 'UL' LIMIT 5;"

try:
    df = pd.read_sql_query(query1, conn)
    print("Query successful!")
    print(f"Results: {len(df)} rows")
    print(df.head())
except Exception as e:
    print(f"Error: {str(e)}")

# Test simpler version
query2 = "SELECT DISTINCT CUCRecipeCode, CUCPlantDescription FROM RecipeExplosion WHERE CUCPlantOwnershipFlag = 'UL' LIMIT 5;"

try:
    df2 = pd.read_sql_query(query2, conn)
    print("\nSimpler query successful!")
    print(f"Results: {len(df2)} rows")
    print(df2.head())
except Exception as e:
    print(f"Simpler query error: {str(e)}")

conn.close()
