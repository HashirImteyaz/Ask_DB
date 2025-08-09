#!/usr/bin/env python3
"""
Create sample data to demonstrate the PLM system functionality.
This creates minimal test data for the Specifications and RecipeExplosion tables.
"""

import pandas as pd
import os
from pathlib import Path

def create_sample_data():
    """Create sample CSV files for testing the system."""
    
    # Create DATA directory if it doesn't exist
    data_dir = Path("DATA")
    data_dir.mkdir(exist_ok=True)
    
    # Sample Specifications data (both recipes and ingredients)
    specifications_data = [
        # Recipes (CUC - Consumer Unit Content)
        {"SpecCode": "R001", "SpecDescription": "Italian Tomato Pasta Sauce", "SpecGroupCode": "CUC", "SpecStatus": "Active"},
        {"SpecCode": "R002", "SpecDescription": "Chicken Noodle Soup Mix", "SpecGroupCode": "CUC", "SpecStatus": "Active"},
        {"SpecCode": "R003", "SpecDescription": "Vegetable Curry Powder", "SpecGroupCode": "CUC", "SpecStatus": "Active"},
        {"SpecCode": "R004", "SpecDescription": "Beef Bouillon Cube", "SpecGroupCode": "CUC", "SpecStatus": "Active"},
        {"SpecCode": "R005", "SpecDescription": "French Onion Soup Mix", "SpecGroupCode": "CUC", "SpecStatus": "Active"},
        
        # Ingredients (ING)
        {"SpecCode": "I001", "SpecDescription": "Tomatoes", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I002", "SpecDescription": "Basil", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I003", "SpecDescription": "Garlic", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I004", "SpecDescription": "Chicken Extract", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I005", "SpecDescription": "Noodles", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I006", "SpecDescription": "Carrots", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I007", "SpecDescription": "Turmeric", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I008", "SpecDescription": "Cumin", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I009", "SpecDescription": "Coriander", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I010", "SpecDescription": "Beef Extract", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I011", "SpecDescription": "Salt", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I012", "SpecDescription": "Onions", "SpecGroupCode": "ING", "SpecStatus": "Active"},
        {"SpecCode": "I013", "SpecDescription": "Celery", "SpecGroupCode": "ING", "SpecStatus": "Active"},
    ]
    
    # Sample RecipeExplosion data (recipe-ingredient relationships)
    recipe_explosion_data = [
        # Italian Tomato Pasta Sauce (R001)
        {"CUCSpecCode": "R001", "INGSpecCode": "I001", "Ing2CUC_PercentageContribution": 60.0, "Ing2CUC_QuantityContribution": 300.0},
        {"CUCSpecCode": "R001", "INGSpecCode": "I002", "Ing2CUC_PercentageContribution": 5.0, "Ing2CUC_QuantityContribution": 25.0},
        {"CUCSpecCode": "R001", "INGSpecCode": "I003", "Ing2CUC_PercentageContribution": 3.0, "Ing2CUC_QuantityContribution": 15.0},
        {"CUCSpecCode": "R001", "INGSpecCode": "I011", "Ing2CUC_PercentageContribution": 2.0, "Ing2CUC_QuantityContribution": 10.0},
        
        # Chicken Noodle Soup Mix (R002)
        {"CUCSpecCode": "R002", "INGSpecCode": "I004", "Ing2CUC_PercentageContribution": 35.0, "Ing2CUC_QuantityContribution": 175.0},
        {"CUCSpecCode": "R002", "INGSpecCode": "I005", "Ing2CUC_PercentageContribution": 40.0, "Ing2CUC_QuantityContribution": 200.0},
        {"CUCSpecCode": "R002", "INGSpecCode": "I006", "Ing2CUC_PercentageContribution": 15.0, "Ing2CUC_QuantityContribution": 75.0},
        {"CUCSpecCode": "R002", "INGSpecCode": "I013", "Ing2CUC_PercentageContribution": 8.0, "Ing2CUC_QuantityContribution": 40.0},
        {"CUCSpecCode": "R002", "INGSpecCode": "I011", "Ing2CUC_PercentageContribution": 2.0, "Ing2CUC_QuantityContribution": 10.0},
        
        # Vegetable Curry Powder (R003)
        {"CUCSpecCode": "R003", "INGSpecCode": "I007", "Ing2CUC_PercentageContribution": 30.0, "Ing2CUC_QuantityContribution": 150.0},
        {"CUCSpecCode": "R003", "INGSpecCode": "I008", "Ing2CUC_PercentageContribution": 25.0, "Ing2CUC_QuantityContribution": 125.0},
        {"CUCSpecCode": "R003", "INGSpecCode": "I009", "Ing2CUC_PercentageContribution": 20.0, "Ing2CUC_QuantityContribution": 100.0},
        {"CUCSpecCode": "R003", "INGSpecCode": "I003", "Ing2CUC_PercentageContribution": 15.0, "Ing2CUC_QuantityContribution": 75.0},
        {"CUCSpecCode": "R003", "INGSpecCode": "I011", "Ing2CUC_PercentageContribution": 10.0, "Ing2CUC_QuantityContribution": 50.0},
        
        # Beef Bouillon Cube (R004)
        {"CUCSpecCode": "R004", "INGSpecCode": "I010", "Ing2CUC_PercentageContribution": 45.0, "Ing2CUC_QuantityContribution": 225.0},
        {"CUCSpecCode": "R004", "INGSpecCode": "I011", "Ing2CUC_PercentageContribution": 30.0, "Ing2CUC_QuantityContribution": 150.0},
        {"CUCSpecCode": "R004", "INGSpecCode": "I012", "Ing2CUC_PercentageContribution": 15.0, "Ing2CUC_QuantityContribution": 75.0},
        {"CUCSpecCode": "R004", "INGSpecCode": "I003", "Ing2CUC_PercentageContribution": 10.0, "Ing2CUC_QuantityContribution": 50.0},
        
        # French Onion Soup Mix (R005)
        {"CUCSpecCode": "R005", "INGSpecCode": "I012", "Ing2CUC_PercentageContribution": 50.0, "Ing2CUC_QuantityContribution": 250.0},
        {"CUCSpecCode": "R005", "INGSpecCode": "I010", "Ing2CUC_PercentageContribution": 25.0, "Ing2CUC_QuantityContribution": 125.0},
        {"CUCSpecCode": "R005", "INGSpecCode": "I011", "Ing2CUC_PercentageContribution": 20.0, "Ing2CUC_QuantityContribution": 100.0},
        {"CUCSpecCode": "R005", "INGSpecCode": "I003", "Ing2CUC_PercentageContribution": 5.0, "Ing2CUC_QuantityContribution": 25.0},
    ]
    
    # Create DataFrames
    df_specs = pd.DataFrame(specifications_data)
    df_recipe_explosion = pd.DataFrame(recipe_explosion_data)
    
    # Save to CSV files
    specs_file = data_dir / "SpecsForLLM_20250802.csv"
    recipe_file = data_dir / "RecipeExplosionForLLM_20250731_v2.csv"
    
    df_specs.to_csv(specs_file, index=False)
    df_recipe_explosion.to_csv(recipe_file, index=False)
    
    print(f"✅ Created sample data files:")
    print(f"   - {specs_file} ({len(df_specs)} specifications)")
    print(f"   - {recipe_file} ({len(df_recipe_explosion)} recipe relationships)")
    print(f"\nSample recipes created:")
    for _, row in df_specs[df_specs['SpecGroupCode'] == 'CUC'].iterrows():
        print(f"   - {row['SpecDescription']}")
    
    print(f"\nTo populate the database, run:")
    print(f"   python scripts/data_setup.py")

if __name__ == "__main__":
    create_sample_data()
