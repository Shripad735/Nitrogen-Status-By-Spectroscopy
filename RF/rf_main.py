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
    train_file_name='train_data.parquet',
    validation_file_name='validation_data.parquet',
    test_file_name='test_data.parquet'
    )

# Initialize and run RFModel
rf_model = RFModel(dataset, param_grid)
results = {
                'best_params': self.best_params,
                'cv_avg_rmse': self.cv_avg_rmse,
                'test_rmse': self.test_rmse
            }

print("Model training and evaluation complete.")
print(f"Best Hyperparameters: {results['best_params']}")
print(f"Cross-Validation Average RMSE: {results['cv_avg_rmse']}")
print(f"Test RMSE: {results['test_rmse']}")