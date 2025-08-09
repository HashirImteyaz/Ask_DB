import pandas as pd
from sqlalchemy import create_engine

# A simple DataFrame to save
df_test = pd.DataFrame({'id': [1, 2], 'value': ['a', 'b']})

# The exact path you want to use
recp_explo_file = "C:\\Users\\ShyamSunderIyer\\Downloads\\RecipeExplosionForLLM_20250731_v2.csv"
spec_file = "C:\\Users\\ShyamSunderIyer\\Downloads\\SpecsForLLM_20250802.csv"

# engine_test = create_engine(db_url_test)

# # Save the DataFrame to a table named 'test_table'
# df_test.to_sql('test_table', engine_test, if_exists='replace', index=False)

# print("Test table should be saved. Please check the specified path.")

#READ CSVs and extractions
df_spec = pd.read_csv(spec_file)
columns_to_remove = ['DUSpecCode', 'DURecipeCode', 'DUMaterialCode', 'DUPlantCode', 'DUPlantDescription', 
                     'DUPlantOwnershipFlag', 'DUPlantCountryCode', 'DUPlantCountryName', 'DUPlantBUCode', 
                     'DUPlantBUName', 'ExplosionRecipePathValue', 'ExplosionSpecPathValue', 'ExplosionMaterialPathValue']
df_explo =  pd.read_csv(recp_explo_file).drop(columns=columns_to_remove)

# print("SPEC DTYPES", df_spec.dtypes)
# print("EXPLO DTYPES", df_explo.dtypes)
valid_spec_codes = df_spec['SpecCode']
cuc_is_valid = df_explo['CUCSpecCode'].isin(valid_spec_codes)
ing_is_valid = df_explo['INGSpecCode'].isin(valid_spec_codes)
df_explo_filtered = df_explo[cuc_is_valid & ing_is_valid].reset_index(drop=True)
if df_explo_filtered.empty:
    print("Nothing is there after filtering")
    exit()

print("Data filtered successfully.")
print(f"Number of rows after filtering: {len(df_explo_filtered)}")

tables = {
    "Specifications": df_spec,
    "RecipeExplosion": df_explo_filtered
    }

# UPLOADING TO DB

from sqlalchemy import create_engine
import numpy as np
from sqlalchemy import MetaData, inspect
from src.core.data_processing.utils import create_table_from_dataframe, drop_all_tables

DB_URL = "sqlite:///C:/Users/ShyamSunderIyer/Documents/test.db"
NEW_ENGINE = create_engine(DB_URL)
meta = MetaData()
drop_all_tables(NEW_ENGINE)
for table_name, df in tables.items():
    df = df.replace({pd.NA: None, np.nan: None})
    df.to_sql(
        name=table_name, con=NEW_ENGINE,
        if_exists='append',  # 'fail', 'replace', or 'append'
        index=False         # Set to True if you want to save the DataFrame index as a column
    )
    print(f"Successfully inserted data into '{table_name}'.")

# CHECKING PART
inspector = inspect(NEW_ENGINE)
table_names = inspector.get_table_names()
if not table_names:
    print("Nothing got saved")
    exit()
for name in table_names:
    columns = inspector.get_columns(name)
    print(name, columns)
