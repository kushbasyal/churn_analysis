from paths import raw_data_path,clean_data_path
from loader import load_data
from cleaner import inspect_data, clean_data
from clustering import find_cluster, visualise_cluster

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

    # Run clustering evaluation on selected columns
    print("\n====== RUNNING CLUSTERING ANALYSIS ======")
    find_cluster(clean_df[["age", "tenure"]])

    # 2. Visualize clusters with chosen k (e.g., k = 3)
    visualise_cluster(clean_df[["age", "balance"]], k=3)

if __name__ == '__main__':
    main()