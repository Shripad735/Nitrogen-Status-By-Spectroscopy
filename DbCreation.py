import pandas as pd


def remove_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def remove_rows_with_negatives(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_check = ['ST_Value', 'SC_Value', 'N_Value']
    return df[(df[columns_to_check] >= 0).all(axis=1)]


def main():
    # Read the merged_data.parquet file into a DataFrame
    parquet_file_path = 'merged_data.parquet'
    merged_df = pd.read_parquet(parquet_file_path)

    # Remove rows with null values and save to a new parquet file
    merged_df_without_nulls = remove_rows_with_nulls(merged_df)
    parquet_file_path_without_nulls = 'merged_data_without_nulls.parquet'
    merged_df_without_nulls.to_parquet(parquet_file_path_without_nulls, index=False)

    # Remove rows with negative values and save to a new parquet file
    merged_df_with_positive_values = remove_rows_with_negatives(merged_df)
    parquet_file_path_with_positive_values = 'merged_data_with_positive_values.parquet'
    merged_df_with_positive_values.to_parquet(parquet_file_path_with_positive_values, index=False)

    print(f"Number of rows in merged_df: {len(merged_df)}")
    print(f"Number of rows in merged_df_without_nulls: {len(merged_df_without_nulls)}")
    print(f"Number of rows in merged_df_with_positive_values: {len(merged_df_with_positive_values)}")


if __name__ == "__main__":
    main()
