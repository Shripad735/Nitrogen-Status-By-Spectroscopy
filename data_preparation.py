import pandas as pd
from sklearn.model_selection import train_test_split
from constants_config import ColumnName, CropPart, IDComponents, DATA_FOLDER


def filter_leaf_samples(df: pd.DataFrame) -> pd.DataFrame:
    # Extract crop and part from the ID column and convert to lowercase
    df.loc[:, IDComponents.crop.value] = df[ColumnName.id.value].str[:3].str.lower()
    df.loc[:, IDComponents.part.value] = df[ColumnName.id.value].str[6:9].str.lower()

    # Update incorrect classifier 'ea2' to 'lea'
    df.loc[df[IDComponents.part.value] == 'ea2', IDComponents.part.value] = CropPart.leaf.value

    # Filter the DataFrame to include only rows representing leaf samples
    df_leaf_samples = df[df[IDComponents.part.value] == CropPart.leaf.value]
    return df_leaf_samples


def analyze_explained_variables_per_crop(df: pd.DataFrame, include_negatives: bool):
    # Count the number of n_values, sc_value, and st_values for each crop and part
    crop_part_value_counts = df.groupby([IDComponents.crop.value, IDComponents.part.value]).agg({
        ColumnName.n_value.value: 'count',
        ColumnName.sc_value.value: 'count',
        ColumnName.st_value.value: 'count'
    }).reset_index()

    for crop in df[IDComponents.crop.value].unique():
        crop_data = crop_part_value_counts[crop_part_value_counts[IDComponents.crop.value] == crop]
        parts = crop_data[IDComponents.part.value]
        n_values_counts = crop_data[ColumnName.n_value.value]
        sc_values_counts = crop_data[ColumnName.sc_value.value]
        st_values_counts = crop_data[ColumnName.st_value.value]

        # Print the parameters
        print(f"\nCrop: {crop}")
        for part, n_val, sc_val, st_val in zip(parts, n_values_counts, sc_values_counts, st_values_counts):
            if include_negatives:
                negative_counts = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.part.value] == part)][
                    [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]].lt(0).sum()
                negative_counts = {k: v for k, v in negative_counts.items() if v > 0}
                if negative_counts:
                    print(
                        f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                        f"{ColumnName.st_value.value}: {st_val}, Negative values: {negative_counts}")
                else:
                    print(f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                          f"{ColumnName.st_value.value}: {st_val}")
            else:
                print(f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                      f"{ColumnName.st_value.value}: {st_val}")


def analyze_crops(df: pd.DataFrame, include_negatives: bool = True):
    unique_crops = df[IDComponents.crop.value].unique()
    print(f"\nNumber of different crops: {len(unique_crops)}")

    crop_counts = df[IDComponents.crop.value].value_counts()
    print("\nNumber of rows for each crop:")

    for crop, count in crop_counts.items():
        print(f"Crop: {crop}, Rows: {count}")
    print(f'\nTotal amount of rows: {df.shape[0]}')

    analyze_explained_variables_per_crop(df, include_negatives)


def remove_observations_with_missing_explained_variables(df: pd.DataFrame) -> pd.DataFrame:
    explained_variables = [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]
    return df.dropna(subset=explained_variables)


def replace_negative_values_with_mean_or_median(df: pd.DataFrame, method: str) -> pd.DataFrame:
    # Median is robust to outliers and provides a better measure of central tendency for skewed distributions,
    # offering a more accurate representation of the typical value in datasets with outliers.
    explained_variables = [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]

    for column in explained_variables:
        if method == 'mean':
            replacement_value = df[df[column] >= 0][column].mean()
        elif method == 'median':
            replacement_value = df[df[column] >= 0][column].median()
        else:
            raise ValueError("Method must be 'mean' or 'median'")

        # Replace only the negative values
        negative_values = df[df[column] < 0][column]
        negative_count = negative_values.shape[0]
        if negative_count > 0:
            df.loc[df[column] < 0, column] = replacement_value
            example_negative_value = negative_values.iloc[0]
            print(f"Column '{column}' had {negative_count} negative values. "
                  f"Method: {method}. Replacement value: {replacement_value}. "
                  f"Example of negative value: {example_negative_value}")

    return df


def split_dataset(df: pd.DataFrame, train_size: float = 0.7, validation_size: float = 0.15, test_size: float = 0.15,
                  random_state: int = 42):
    if train_size + validation_size + test_size != 1.0:
        raise ValueError("The sum of train_size, validation_size, and test_size must be 1.0")

    # Split the data into train and temp sets
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)

    # Calculate the proportion of validation and test sizes relative to the temp set
    validation_proportion = validation_size / (validation_size + test_size)

    # Split the temp set into validation and test sets
    validation_df, test_df = train_test_split(temp_df, train_size=validation_proportion, random_state=random_state)

    return train_df, validation_df, test_df


def save_and_print_dataset_splits(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame,
                                  folder_path: str = DATA_FOLDER):
    # Print the size of each file
    print(f"\nTrain set size: {train_df.shape[0]} rows")
    print(f"Validation set size: {validation_df.shape[0]} rows")
    print(f"Test set size: {test_df.shape[0]} rows")

    # Save the datasets to parquet files in the specified folder
    train_file_path = f'{folder_path}/train_data.parquet'
    validation_file_path = f'{folder_path}/validation_data.parquet'
    test_file_path = f'{folder_path}/test_data.parquet'

    train_df.to_parquet(train_file_path)
    validation_df.to_parquet(validation_file_path)
    test_df.to_parquet(test_file_path)


def main():
    # Read the merged_data.parquet file into a DataFrame
    parquet_file_path = f'{DATA_FOLDER}/merged_data.parquet'
    df = pd.read_parquet(parquet_file_path)

    # Filter the DataFrame to include only rows representing leaf samples
    df = filter_leaf_samples(df)

    analyze_crops(df, include_negatives=True)

    print("\n\nRemoving observations with missing explained variables")
    df = remove_observations_with_missing_explained_variables(df)

    analyze_explained_variables_per_crop(df, include_negatives=True)

    print("\nHandling observations with negative values")
    df = replace_negative_values_with_mean_or_median(df, method='median')

    analyze_explained_variables_per_crop(df, include_negatives=True)

    # Split the dataset into train, validation, and test sets
    train_df, validation_df, test_df = split_dataset(df)

    # Save the datasets to parquet files and print their sizes
    save_and_print_dataset_splits(train_df, validation_df, test_df)


if __name__ == "__main__":
    main()
