import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_csv(file_path):
    df = pd.read_csv(file_path)
    print("Original Data:")
    print(df.head())

    if 'Rating' not in df.columns:
        print("The column 'Rating' was not found in the CSV file.")
        return

    rating_col = df['Rating']

    not_null_mask = rating_col.notnull()
    scaler = MinMaxScaler()
    normalized_values = rating_col.copy()
    normalized_values[not_null_mask] = scaler.fit_transform(rating_col[not_null_mask].values.reshape(-1, 1)).flatten()

    df['Rating'] = normalized_values

    output_file = 'normalized_' + file_path
    df.to_csv(output_file, index=False)
    print(f"\nNormalized 'Rating' column saved to '{output_file}'")

def main():
    file_path = input("Enter the path to your CSV file: ")
    normalize_csv(file_path)

if __name__ == "__main__":
    main()
