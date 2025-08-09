import pyodbc
import os

# Database connection details
# ServerName: plmsqlppd.database.windows.net
# Authentication: SQL Server Authentication
# Username: plmadmin@plmsqlppd
# Password: POI@5467
# Database Name: ulplmsqlppd
 
 
 
server_name = 'plmsqlppd.database.windows.net'
database_name = 'ulplmsqlppd'
username = 'plmadmin@plmsqlppd'
password = 'POI@5467'

# Construct the connection string
# The driver may need to be adjusted based on your system.
# 'ODBC Driver 17 for SQL Server' is a common choice.
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

        # Create a cursor object to execute queries
        with cnxn.cursor() as cursor:
            # Example: Execute a simple query to get the database name
            cursor.execute("SELECT DB_NAME()")
            row = cursor.fetchone()
            if row:
                print(f"Current database name is: {row[0]}")

except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f"Error connecting to the database. SQL State: {sqlstate}")
    print(ex)
