from paths import raw_data_path, clean_data_path
from loader import load_data

def inspect_data(df):

    print("\n===== Shape =====")
    print(df.shape)

    print("\n===== Data Types =====")
    print(df.dtypes)

    print("\n===== Info =====")
    df.info()

    print("\n===== Describe =====")
    print(df.describe())

    print("\n===== Null Values =====")
    print(df.isna().sum())

    print("\n===== Duplicates =====")
    print(df.duplicated().sum())

    print("\n===== Numerical Columns =====")
    print(df.select_dtypes(include="number").columns)

    print("\n===== Categorical Columns =====")
    print(df.select_dtypes(include=["object", "category", "bool"]).columns)


def clean_data(df):

    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Numerical columns
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        df[col] = df[col].str.strip()

        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df
