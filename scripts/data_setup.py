# data_setup.py - Configurable data processing script

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData
from src.core.data_processing.utils import drop_all_tables
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_data_from_config(config_path: str = None):
    """Setup database from configuration file or environment variables."""
    
    # Configuration with defaults
    config = {
        'recipe_explosion_file': os.getenv('RECIPE_EXPLOSION_FILE', 'DATA/RecipeExplosionForLLM_20250731_v2.csv'),
        'specifications_file': os.getenv('SPECIFICATIONS_FILE', 'DATA/SpecsForLLM_20250802.csv'),
        'output_db_url': os.getenv('OUTPUT_DB_URL', 'sqlite:///DATA/plm_updated.db'),
        'columns_to_remove': [
            'DUSpecCode', 'DURecipeCode', 'DUMaterialCode', 'DUPlantCode', 
            'DUPlantDescription', 'DUPlantOwnershipFlag', 'DUPlantCountryCode', 
            'DUPlantCountryName', 'DUPlantBUCode', 'DUPlantBUName', 
            'ExplosionRecipePathValue', 'ExplosionSpecPathValue', 'ExplosionMaterialPathValue'
        ]
    }
    
    # Validate input files exist
    for file_key in ['recipe_explosion_file', 'specifications_file']:
        file_path = Path(config[file_key])
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        logger.info(f"Found {file_key}: {file_path}")
    
    # Load and process data
    logger.info("Loading specification data...")
    df_spec = pd.read_csv(config['specifications_file'])
    logger.info(f"Loaded {len(df_spec)} specification records")
    
    logger.info("Loading recipe explosion data...")
    df_explo = pd.read_csv(config['recipe_explosion_file'])
    
    # Remove unnecessary columns if they exist
    existing_cols_to_remove = [col for col in config['columns_to_remove'] if col in df_explo.columns]
    if existing_cols_to_remove:
        df_explo = df_explo.drop(columns=existing_cols_to_remove)
        logger.info(f"Removed {len(existing_cols_to_remove)} unnecessary columns")
    
    logger.info(f"Loaded {len(df_explo)} recipe explosion records")
    
    # Data validation and filtering
    logger.info("Validating data integrity...")
    valid_spec_codes = set(df_spec['SpecCode'])
    
    initial_count = len(df_explo)
    cuc_valid = df_explo['CUCSpecCode'].isin(valid_spec_codes)
    ing_valid = df_explo['INGSpecCode'].isin(valid_spec_codes)
    df_explo_filtered = df_explo[cuc_valid & ing_valid].reset_index(drop=True)
    
    if df_explo_filtered.empty:
        raise ValueError("No valid records found after filtering. Check data integrity.")
    
    logger.info(f"Filtered data: {initial_count} → {len(df_explo_filtered)} records ({len(df_explo_filtered)/initial_count*100:.1f}% retained)")
    
    # Prepare tables for database insertion
    tables = {
        "Specifications": df_spec,
        "RecipeExplosion": df_explo_filtered
    }
    
    # Database setup
    logger.info(f"Setting up database: {config['output_db_url']}")
    engine = create_engine(config['output_db_url'])
    
    # Clear existing tables
    drop_all_tables(engine)
    logger.info("Cleared existing tables")
    
    # Insert data
    for table_name, df in tables.items():
        # Handle NaN values
        df_clean = df.replace({pd.NA: None, np.nan: None})
        
        df_clean.to_sql(
            name=table_name, 
            con=engine,
            if_exists='append',
            index=False
        )
        logger.info(f"Inserted {len(df_clean)} records into '{table_name}'")
    
    # Verification
    from sqlalchemy import inspect
    inspector = inspect(engine)
    final_tables = inspector.get_table_names()
    
    if not final_tables:
        raise RuntimeError("Database setup failed - no tables created")
    
    logger.info("Database setup completed successfully!")
    logger.info("Tables created:")
    
    for table_name in final_tables:
        columns = inspector.get_columns(table_name)
        logger.info(f"  - {table_name}: {len(columns)} columns")
    
    return engine, tables

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Setup PLM database from CSV files')
    parser.add_argument('--recipe-file', help='Path to recipe explosion CSV file')
    parser.add_argument('--spec-file', help='Path to specifications CSV file')  
    parser.add_argument('--output-db', help='Output database URL')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Override environment variables with command line args
    if args.recipe_file:
        os.environ['RECIPE_EXPLOSION_FILE'] = args.recipe_file
    if args.spec_file:
        os.environ['SPECIFICATIONS_FILE'] = args.spec_file
    if args.output_db:
        os.environ['OUTPUT_DB_URL'] = args.output_db
    
    try:
        engine, tables = setup_data_from_config(args.config)
        logger.info("[PASS] Data setup completed successfully!")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Data setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)