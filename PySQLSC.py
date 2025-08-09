import pyodbc
import os
from dotenv import load_dotenv
load_dotenv()
 
 
 
server_name = os.getenv("SERVER_NAME")
database_name = os.getenv("DB_NAME")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")

connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={server_name};"
    f"DATABASE={database_name};"
    f"UID={username};"
    f"PWD={password};"
)

try:
    # Connect to the database
    with pyodbc.connect(connection_string) as cnxn:
        print("Successfully connected to the database!")
        with cnxn.cursor() as cursor:
            cursor.execute("""SELECT TOP(10) SpecDescription FROM plm.Specifications WHERE AuthorisationGroupCode = 'ZSVR' AND SpecGroupCode = 'CUC'""")
            row = cursor.fetchall()
            if row:
                print(f"Current database name is: {row}")

except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f"Error connecting to the database. SQL State: {sqlstate}")
    print(ex)
