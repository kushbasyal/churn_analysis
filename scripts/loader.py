from paths import raw_data_path, clean_data_path
import pandas as pd
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("===== Data Loaded Sucessfully =====")
        return df
    except FileNotFoundError:
        print("===== File Path Not Found =====")
        return None

if __name__ == '__main__':
    load_data(raw_data_path)