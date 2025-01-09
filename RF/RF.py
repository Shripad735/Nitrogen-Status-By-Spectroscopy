import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os as os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from baseline_for_training.Dataset import Dataset
from baseline_for_training.baseModels import BaseModel
from baseline_for_training.Training_and_Tuning import hyperParameterTuning, CV10
from constants_config import TARGET_VARIABLES, COLOR_PALETTE_FOR_PLSR, AVG_RMSE

class RFModel(BaseModel):
    def __init__(self, dataset: Dataset, param_grid: dict, model_path: str, is_multi_output=True, target_variable_name=None):
            if is_multi_output:
                model = MultiOutputRegressor(RandomForestRegressor())
            else:
                model = RandomForestRegressor()
            super().__init__(dataset, model, param_grid, is_multi_output, target_variable_name)
            self.best_params = None
            self.cv_rmse = None
            self.test_rmse = {}
            self.model_path = model_path




    def run(self):
        self.__tune_hyperparameters()
        self.__cross_validate()
        self.__train_and_evaluate_model()
        self.eval_plot()
        self.__save_test_results()
        self.__save_model()


    def __tune_hyperparameters(self):
        print("Starting hyperparameter tuning...")
        tuning_results = hyperParameterTuning(self, PLSR_Tuning=False)
        self.best_params = min(tuning_results['Avg_RMSE'], key=lambda x: x[1])[0]
        print(f"Best Parameters: {self.best_params}")
        self.model.estimator.set_params(**self.best_params)


    def __cross_validate(self):
        print("Starting cross-validation...")
        self.cv_rmse = CV10(self, n_splits=10)
        # Plot RMSE vs Folds
        plt.figure(figsize=(10, 6))
        plt.plot([i for i in range(1, 11)], self.cv_rmse['Avg_RMSE'], label='Avg_RMSE')
        plt.plot([i for i in range(1, 11)], self.cv_rmse['N_Value'], label='N Value')
        plt.plot([i for i in range(1, 11)], self.cv_rmse['SC_Value'], label='SC Value')
        plt.plot([i for i in range(1, 11)], self.cv_rmse['ST_Value'], label='ST Value')
        plt.xlabel('Folds')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Folds')
        plt.legend()
        plt.savefig('./plots/RMSE_vs_Folds_RF.png')


    def __train_and_evaluate_model(self):
        print("Training on train set...")
        self.model.fit(self.dataset.X_train, self.dataset.Y_train)
        print("Evaluating on test set...")
        rmse_results = self.evaluate()
        self.test_rmse.update(zip(TARGET_VARIABLES, rmse_results))
        self.test_rmse[AVG_RMSE] = np.mean(rmse_results)
        print(f"Test RMSE: {self.test_rmse}")

    def eval_plot(self):
        # plot as hist the test rmse results
        res = self.test_rmse
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(res.keys(), res.values(), color=[COLOR_PALETTE_FOR_PLSR[key] for key in res.keys()])
        labels = ['N_Value', 'SC_Value', 'ST_Value', 'Avg_RMSE']
        for label in labels:
            bars[labels.index(label)].set_label(label)

        ax.set_xlabel('Target Variables', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.legend(fontsize=10)
        fig.savefig('./Plots/Test_RMSE_RF.png', dpi=300, bbox_inches='tight')




        # Plot the predicted vs actual values for each model
        y_hat = self.model.predict(self.dataset.X_test)
        y_test = self.dataset.Y_test.to_numpy()

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        axs[0, 0].scatter(y_test[:, 0], y_hat[:, 0])
        axs[0, 0].set_title('N Value')
        axs[0, 0].set_xlabel('Actual')
        axs[0, 0].set_ylabel('Predicted')

        axs[0, 1].scatter(y_test[:, 1], y_hat[:, 1])
        axs[0, 1].set_title('SC Value')
        axs[0, 1].set_xlabel('Actual')
        axs[0, 1].set_ylabel('Predicted')

        axs[1, 0].scatter(y_test[:, 2], y_hat[:, 2])
        axs[1, 0].set_title('ST Value')
        axs[1, 0].set_xlabel('Actual')

        # Disable last plot
        axs[1, 1].axis('off')
        plt.savefig('./Plots/Predicted_vs_Actual_Values_RF.png')

        # Plot the residuals against the predicted values for the multi output PLSR model

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        residuals = y_test - y_hat

        axs[0, 0].scatter(y_hat[:, 0], residuals[:, 0])
        axs[0, 0].set_title('N Value')
        axs[0, 0].set_xlabel('Predicted')
        axs[0, 0].set_ylabel('Residual')
        axs[0, 0].axhline(y=0, color='r', linestyle='-')

        axs[0, 1].scatter(y_hat[:, 1], residuals[:, 1])
        axs[0, 1].set_title('SC Value')
        axs[0, 1].set_xlabel('Predicted')
        axs[0, 1].set_ylabel('Residual')
        axs[0, 1].axhline(y=0, color='r', linestyle='-')

        axs[1, 0].scatter(y_hat[:, 2], residuals[:, 2])
        axs[1, 0].set_title('ST Value')
        axs[1, 0].set_xlabel('Predicted')
        axs[1, 0].axhline(y=0, color='r', linestyle='-')

        # Disable last plot
        axs[1, 1].axis('off')
        plt.savefig('./Plots/Residuals_vs_Predicted_Values_RF.png')

    def __save_model(self):
        print("Saving model...")
        joblib.dump(self, os.path.join(self.model_path, 'rf_model.pkl'))
        print("Model saved.")

    def __save_test_results(self):
        # save test results as json file
        print("Saving test results...")
        with open(os.path.join(self.model_path, 'test_results.json'), 'w') as f:
            json.dump(self.test_rmse, f)


    def load_model(self):
        print("Loading model...")
        model = joblib.load(os.path.join(self.model_path, 'rf_model.pkl'))
        print("Model loaded")
        return model



