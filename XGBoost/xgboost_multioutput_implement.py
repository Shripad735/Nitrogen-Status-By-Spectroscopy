import os
import json
import joblib
import random
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants_config import ColumnName
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


class XGBoostMultiOutput:
    def __init__(self, n_splits=10, save_dir="models/"):
        self.val_data = None
        self.train_data = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.models = None
        self.best_params = None

    def load_data(self, train_path, val_path):
        self.train_data = pd.read_parquet(f'../{train_path}')
        self.val_data = pd.read_parquet(f'../{val_path}')

    def preprocess_data(self, dataset="train"):
        feature_columns = [col for col in self.train_data.columns if
                           col not in [ColumnName.id, ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]]
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[[ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[[ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]]
        return X, y

    def get_param_grid(self):
        # # Define the hyperparameter grid - 4*6*4*3*3*4*4 = 13824 configurations
        # param_grid = {
        #     "learning_rate": [0.01, 0.05, 0.1, 0.2],
        #     "max_depth": [3, 4, 5, 6, 7, 8],
        #     "n_estimators": [50, 100, 200, 300],
        #     "subsample": [0.6, 0.8, 1.0],
        #     "colsample_bytree": [0.6, 0.8, 1.0],
        #     "gamma": [0, 0.1, 0.2, 0.3],
        #     "reg_lambda": [1, 1.5, 2, 3]
        # }
        # Define a smaller hyperparameter grid with just 2 configurations
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [50, 100],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0],
            "reg_lambda": [1]
        }
        keys, values = zip(*param_grid.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return configurations

    def train_and_evaluate(self, params, X_train, y_train, X_val, y_val):
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        # multioutput='raw_values' parameter, returns an array of RMSE values, one for each target variable.
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val, multioutput='raw_values')).mean()
        return rmse, model

    def find_best_configuration(self, X_train, y_train, X_val, y_val):
        param_grid = self.get_param_grid()
        best_rmse = float("inf")
        best_params = None
        best_model = None

        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            rmse, model = self.train_and_evaluate(params, X_train, y_train, X_val, y_val)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model

        self.best_params = best_params
        self.models = best_model
        return best_rmse, best_params

    def cross_validate(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        rmses = []

        for train_index, val_index in tqdm(kf.split(X), desc="Cross-validation"):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            rmse, _ = self.train_and_evaluate(self.best_params, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            rmses.append(rmse)

        return np.mean(rmses)

    def plot_learning_curve(self, model):
        results = model.estimators_[0].evals_result()
        epochs = len(results["validation_0"]["rmse"])
        x_axis = range(0, epochs)

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results["validation_0"]["rmse"], label="Train")
        plt.plot(x_axis, results["validation_1"]["rmse"], label="Validation")
        plt.title("Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, "learning_curve.png"))
        plt.show()

    def plot_feature_importance(self, model):
        importance = model.estimators_[0].get_booster().get_score(importance_type="weight")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        keys, values = zip(*sorted_importance)

        plt.figure(figsize=(12, 6))
        plt.bar(keys[:20], values[:20])  # Top 20 features
        plt.xticks(rotation=90)
        plt.title("Feature Importance")
        plt.xlabel("Feature")
        plt.ylabel("Score")
        plt.savefig(os.path.join(self.save_dir, "feature_importance.png"))
        plt.show()

    def save_model(self, model, filename):
        joblib.dump(model, os.path.join(self.save_dir, filename))

    def save_best_params(self):
        with open(os.path.join(self.save_dir, "best_params.json"), "w") as f:
            json.dump(self.best_params, f)

    def run(self, train_path, val_path, test_path):
        self.load_data(train_path, val_path)
        X_train, y_train = self.preprocess_data(dataset="train")
        X_val, y_val = self.preprocess_data(dataset="val")

        print("Finding best configuration...")
        best_rmse, best_params = self.find_best_configuration(X_train, y_train, X_val, y_val)
        print(f"Best RMSE: {best_rmse}, Best Params: {best_params}")

        print("Cross-validating with best configuration...")
        mean_rmse = self.cross_validate(X_train, y_train)
        print(f"Mean RMSE from CV: {mean_rmse}")

        print("Training final model with best configuration...")
        final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
        final_model.fit(X_train, y_train)

        print("Evaluating final model on validation set...")
        y_pred_val = final_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val, multioutput='raw_values')).mean()
        print(f"Validation RMSE: {val_rmse}")

        print("Evaluating final model on test set...")
        test_data = pd.read_parquet(f'../{test_path}')
        X_test = test_data[[col for col in test_data.columns if
                            col not in [ColumnName.id, ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]]]
        y_test = test_data[[ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]]
        y_pred_test = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput='raw_values')).mean()
        print(f"Test RMSE: {test_rmse}")

        self.save_model(final_model, "final_model.pkl")
        self.save_best_params()
        self.plot_learning_curve(final_model)
        self.plot_feature_importance(final_model)
