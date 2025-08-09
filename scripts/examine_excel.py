# examine_excel.py - Examine the evaluation Excel file

import pandas as pd
import os

def examine_evaluation_file():
    """Examine the evaluation_questions.xlsx file."""
    excel_file = "evaluation/evaluation_questions.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"File not found: {excel_file}")
        return
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)
        print(f"Excel file loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values per column:")
        print(df.isnull().sum())
        
        # Save as JSON for easier examination
        df.to_json("evaluation/evaluation_questions.json", orient="records", indent=2)
        print(f"\nSaved as JSON for reference: evaluation/evaluation_questions.json")
        
        return df
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

if __name__ == "__main__":
    df = examine_evaluation_file()