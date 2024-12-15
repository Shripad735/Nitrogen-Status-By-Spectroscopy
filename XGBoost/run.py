from xgboost_implement import XGBoostRegressorCV
from constants_config import DATA_FOLDER


def main():
    # Initialize and load data
    xgb_cv = XGBoostRegressorCV()
    xgb_cv.load_data(f"../{DATA_FOLDER}/train_data.parquet", f"../{DATA_FOLDER}/validation_data.parquet")

    # Perform hyperparameter tuning, training, and validation
    results = xgb_cv.perform_cv_train_with_tuning()

    # Display validation results
    print("\nValidation Results:")
    for target, metrics in results.items():
        print(f"{target} - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R²']:.4f}")

    # Plot feature importance for N_value
    xgb_cv.plot_feature_importance("N_value")


if __name__ == "__main__":
    main()
