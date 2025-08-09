# data_processing/utils.py

import os
import re
import pandas as pd
import numpy as np
from sqlalchemy import Table, Column, MetaData, Integer, Float, String, Boolean, Text, text
from sqlalchemy.engine import Engine  # <-- Import Engine for type hinting
from fastapi import UploadFile         # <-- Import UploadFile for type hinting
from typing import List


def sanitize_name(name: str) -> str:
    """Sanitizes a string to be a valid table or column name."""
    return re.sub(r'\W+', '_', str(name)).strip('_') or "unnamed"

def infer_sqlalchemy_type(series: pd.Series):
    """Infers the best SQLAlchemy column type from a pandas Series."""
    if pd.api.types.is_integer_dtype(series): return Integer
    if pd.api.types.is_float_dtype(series): return Float
    if pd.api.types.is_bool_dtype(series): return Boolean
    if pd.api.types.is_datetime64_any_dtype(series): return String
    return Text

def drop_all_tables(engine: Engine): # <-- Add type hint
    """Drops all tables from the database."""
    try:
        meta = MetaData()
        meta.reflect(bind=engine)
        for table in reversed(meta.sorted_tables):
            with engine.connect() as conn:
                with conn.begin() as trans:
                    try:
                        conn.execute(text(f'DROP TABLE IF EXISTS "{table.name}"'))
                        trans.commit()
                    except Exception as e:
                        print(f"Could not drop table {table.name}: {e}")
                        trans.rollback()
        print("All tables dropped successfully.")
    except Exception as e:
        print(f"Failed to drop tables: {e}")

def deduplicate_columns(columns: List[str]) -> List[str]: # <-- Add type hint
    """Ensures all column names are unique by appending suffixes if needed."""
    seen: dict = {}
    new_cols: List[str] = []
    for col in columns:
        base = sanitize_name(col)
        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
    return new_cols

def create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine: Engine, metadata_obj: MetaData): # <-- Add type hints
    """Creates a SQL table from a pandas DataFrame, handling data types."""
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    columns = [Column(col, infer_sqlalchemy_type(df[col])) for col in df.columns]
    table = Table(table_name, metadata_obj, *columns)
    metadata_obj.create_all(engine)

    with engine.begin() as conn:
        rows = df.replace({np.nan: None}).to_dict(orient="records")
        if rows:
            conn.execute(table.insert(), rows)

def upload_files_to_db(file_list: List[UploadFile], engine: Engine):
    """Processes a list of uploaded files (Excel/CSV) and loads them into the database."""
    metadata = MetaData()
    
    drop_all_tables(engine)

    for file in file_list:
        try:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file.file)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file.file)
            else:
                continue

            table_name = sanitize_name(os.path.splitext(file.filename)[0])
            df.columns = deduplicate_columns(list(df.columns))

            create_table_from_dataframe(df, table_name, engine, metadata)
            print(f"Successfully processed and loaded '{file.filename}' into table '{table_name}'.")

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            raise e