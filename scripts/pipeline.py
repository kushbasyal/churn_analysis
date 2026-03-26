import os
import json
import pandas as pd

# Get project root

# Folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Project root (parent of the script folder)

root_dir = os.path.dirname(script_dir)

# Load config.json

config_file = os.path.join(root_dir, "config.json")
with open(config_file, "r") as f:
    config = json.load(f)

def extract():
    """Read raw data"""
    raw_path = os.path.join(root_dir, config['paths']['customer_churn_dataset'])
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    return df
def transform(df):
    """Basic cleaning"""
    print("Transforming data: dropping duplicates and missing values")
    df = df.drop_duplicates()
    df = df.dropna()
    return df
def load(df):
    """Save cleaned data"""
    clean_path = os.path.join(root_dir, config['paths']['clean_customer_data'])
    df.to_csv(clean_path, index=False)
    print(f"Saved cleaned data to: {clean_path}")

def run_pipeline():
    """Run the full ETL pipeline"""
    df = extract()
    df_clean = transform(df)
    load(df_clean)
    print("Pipeline complete!")
    return df_clean

if __name__ == "__main__":
    run_pipeline()