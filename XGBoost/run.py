import os
from constants_config import DATA_FOLDER_PATH
from XGBoost.xgboost_multioutput_implement import XGBoostMultiOutput


def main():
    train_path = f"{DATA_FOLDER_PATH}/train_data.parquet"
    val_path = f"{DATA_FOLDER_PATH}/validation_data.parquet"
    test_path = f"{DATA_FOLDER_PATH}/test_data.parquet"

    # Ensure the directory exists
    os.makedirs(DATA_FOLDER_PATH, exist_ok=True)

    xgb_multi_output = XGBoostMultiOutput()
    xgb_multi_output.run(train_path, val_path, test_path)


if __name__ == "__main__":
    main()
