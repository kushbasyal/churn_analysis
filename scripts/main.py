from paths import raw_data_path,clean_data_path
from loader import load_data
from cleaner import inspect_data, clean_data

def main():

    # load raw_data
    raw_df = load_data(raw_data_path)

    # Inspect raw data
    print("====== INSPECT DATA BEFORE CLEANING ======")
    inspect_data(raw_df)

    # Clean data
    clean_df = clean_data(raw_df)

    # Inspect cleaned data
    print("\n====== INSPECT DATA AFTER CLEANING ======")
    inspect_data(clean_df)

    # Save cleaned data
    clean_df.to_csv(clean_data_path, index=False)

if __name__ == '__main__':
    main()