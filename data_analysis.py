import pandas as pd


def analyze_crops(df: pd.DataFrame, include_negatives: bool = True):
    # Extract crop and part from the ID column and convert to lowercase
    df['crop'] = df['ID'].str[:3].str.lower()
    df['part'] = df['ID'].str[6:9].str.lower()

    # Update incorrect classifier 'ea2' to 'lea'
    df.loc[df['part'] == 'ea2', 'part'] = 'lea'

    # Print the number of kinds of crops
    unique_crops = df['crop'].unique()
    print(f"Number of kinds of crops: {len(unique_crops)}")

    # Print the number of rows for each crop
    crop_counts = df['crop'].value_counts()
    print("\nNumber of rows for each crop:")
    for crop, count in crop_counts.items():
        print(f"Crop: {crop}, Rows: {count}")

    # Print the number of rows for each crop and part
    crop_part_counts = df.groupby(['crop', 'part']).size().reset_index(name='counts')
    print("\nNumber of rows for each crop and part:")
    for _, row in crop_part_counts.iterrows():
        print(f"Crop: {row['crop']}, Part: {row['part']}, Rows: {row['counts']}")

    # Count the number of n_values, sc_value, and st_values for each crop and part
    crop_part_value_counts = df.groupby(['crop', 'part']).agg({
        'N_Value': 'count',
        'SC_Value': 'count',
        'ST_Value': 'count'
    }).reset_index()

    for crop in df['crop'].unique():
        crop_data = crop_part_value_counts[crop_part_value_counts['crop'] == crop]
        parts = crop_data['part']
        n_values_counts = crop_data['N_Value']
        sc_values_counts = crop_data['SC_Value']
        st_values_counts = crop_data['ST_Value']

        # Print the parameters
        print(f"\nCrop: {crop}")
        for part, n_val, sc_val, st_val in zip(parts, n_values_counts, sc_values_counts, st_values_counts):
            if include_negatives:
                negative_counts = df[(df['crop'] == crop) & (df['part'] == part)][
                    ['N_Value', 'SC_Value', 'ST_Value']].lt(0).sum()
                negative_counts = {k: v for k, v in negative_counts.items() if v > 0}
                if negative_counts:
                    print(
                        f"Part: {part}, N_Value: {n_val}, SC_Value: {sc_val}, ST_Value: {st_val}, "
                        f"Negative values: {negative_counts}")
                else:
                    print(f"Part: {part}, N_Value: {n_val}, SC_Value: {sc_val}, ST_Value: {st_val}")
            else:
                print(f"Part: {part}, N_Value: {n_val}, SC_Value: {sc_val}, ST_Value: {st_val}")


def main():
    # Read the merged_data.parquet file into a DataFrame
    parquet_file_path = 'merged_data.parquet'
    df = pd.read_parquet(parquet_file_path)

    # Perform the crop analysis with negative values
    analyze_crops(df, include_negatives=True)

    print('\n\n\n\nanalyze_crops_without_negatives\n')

    # Read the merged_data_with_positive_values.parquet file into a DataFrame
    parquet_file_path_positive = 'merged_data_with_positive_values.parquet'
    df_positive = pd.read_parquet(parquet_file_path_positive)

    # Perform the crop analysis without negative values
    analyze_crops(df_positive, include_negatives=False)


if __name__ == "__main__":
    main()
