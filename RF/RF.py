import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from baseline_for_training.Dataset import Dataset
from baseline_for_training.baseModels import BaseModel
from baseline_for_training.Training_and_Tuning import hyperParameterTuning, CV10
from constants_config import TARGET_VARIABLES

class RFModel(BaseModel):
    def __init__(self, dataset, param_grid, is_multi_output=True, target_variable_name=None):
        if is_multi_output:
            model = MultiOutputRegressor(RandomForestRegressor())
        else:
            model = RandomForestRegressor()
        super().__init__(dataset, model, param_grid, is_multi_output, target_variable_name)
        self.best_params = None
        self.cv_rmse = None
        self.test_rmse = None

        self.__run()

    def __run(self):
        self.__preprocess_data()
        self.__tune_hyperparameters()
        self.__cross_validate()
        self.__evaluate_model()

    def __preprocess_data(self):
        # Implement any necessary data preprocessing steps here
        # For example, handling missing values, scaling features, etc.
        pass

    def __tune_hyperparameters(self):
        # print("Starting hyperparameter tuning...")
        # tuning_results = hyperParameterTuning(self, PLSR_Tuning=False)
        # self.best_params = min(tuning_results['Avg_RMSE'], key=lambda x: x[1])[0]
        # print(f"Best Parameters: {self.best_params}")
        self.best_params = {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 20, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
        self.model.estimator.set_params(**self.best_params)

    def __cross_validate(self):
        print("Starting cross-validation...")
        self.cv_rmse = CV10(self, n_splits=10)
        cv_avg_rmse = np.mean(cv_results['Avg_RMSE'])
        print(f"Average RMSE from cross-validation: {cv_avg_rmse}")

    def __evaluate_model(self):
        print("Evaluating on test set...")
        self.test_rmse = self.evaluate()

        print(f"Test RMSE: {self.test_rmse}")


