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
        self.cv_avg_rmse = None
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
        print("Starting hyperparameter tuning...")
        tuning_results = hyperParameterTuning(self, PLSR_Tuning=False)
        self.best_params = min(tuning_results['Avg_RMSE'], key=lambda x: x[1])[0]
        print(f"Best Parameters: {self.best_params}")
        self.model.set_params(**self.best_params)

    def __cross_validate(self):
        print("Starting cross-validation...")
        cv_results = CV10(self, n_splits=10)
        self.cv_avg_rmse = np.mean(cv_results['Avg_RMSE'])
        print(f"Average RMSE from cross-validation: {self.cv_avg_rmse}")

    def __evaluate_model(self):
        print("Evaluating on test set...")
        self.test_rmse = self.evaluate()
        print(f"Test RMSE: {self.test_rmse}")


