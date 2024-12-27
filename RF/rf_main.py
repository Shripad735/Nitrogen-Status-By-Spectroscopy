from RF import RFModel
from baseline_for_training.Dataset import Dataset

param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

# Initialize Dataset
dataset = Dataset(
    train_file_name='train_data_plsr.parquet',
    validation_file_name='validation_data_plsr.parquet',
    test_file_name='test_data_plsr.parquet'
    )

# Initialize and run RFModel
rf_model = RFModel(dataset, param_grid)


print("Model training and evaluation complete.")
print(f"Best Hyperparameters: {results['best_params']}")
print(f"Cross-Validation Average RMSE: {results['cv_avg_rmse']}")
print(f"Test RMSE: {results['test_rmse']}")