import os
import json
import joblib
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from constants_config import TARGET_VARIABLES, NON_FEATURE_COLUMNS, COLOR_PALETTE, MUTLI, TARGET_VARIABLES_WITH_MULTI


class XGBoostMultiOutput:
    def __init__(self, n_splits=2, save_dir='models/', save_figure_dir='figures/'):
        self.val_data = None
        self.train_data = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_figure_dir = save_figure_dir
        os.makedirs(self.save_figure_dir, exist_ok=True)
        self.model = None
        self.best_params = None
        self.train_rmses = {}
        self.val_rmses = {}

    def load_data(self, train_path, val_path):
        self.train_data = pd.read_parquet(train_path)
        self.val_data = pd.read_parquet(val_path)

    def preprocess_data(self, dataset="train"):
        feature_columns = [col for col in self.train_data.columns if col not in NON_FEATURE_COLUMNS]
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[TARGET_VARIABLES]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[TARGET_VARIABLES]
        return X, y

    def get_param_grid(self):
        # Define a smaller hyperparameter grid with just 2 configurations
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [50],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0],
            "reg_lambda": [1],
        }
        keys, values = zip(*param_grid.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return configurations

    def train_and_evaluate(self, params, X_train, y_train, X_val, y_val):
        # Create separate models for each target variable due to different training data that leads to different model
        models = []
        rmses = []
        for i in range(y_train.shape[1]):
            eval_set = [(X_train, y_train.iloc[:, i]), (X_val, y_val.iloc[:, i])]
            model_params = {k: v for k, v in params.items() if k not in ['eval_set', 'eval_metric', 'verbose']}
            model = xgb.XGBRegressor(**model_params, eval_set=eval_set, eval_metric='rmse', verbose=False)
            model.fit(X_train, y_train.iloc[:, i])  # Fit each model individually
            models.append(model)
            y_pred_val = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], y_pred_val))
            rmses.append(rmse)

        # Fourth model: Multi-output regressor
        multi_model = MultiOutputRegressor(xgb.XGBRegressor(**params, eval_metric='rmse', verbose=False))
        multi_model.fit(X_train, y_train)
        y_pred_val_multi = multi_model.predict(X_val)
        multi_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_multi, multioutput='uniform_average'))

        mean_rmse = np.mean(rmses)

        return rmses, mean_rmse, models, multi_model, multi_rmse  # Return individual models

    def save_configurations(self):
        """Save the best configurations (parameters) and their corresponding models."""
        print("Saving best configurations...")
        saved_configs = {}  # Track saved configurations and their folder names
        for target_name, params in self.best_params.items():
            # Create a unique folder name for the parameters
            params_tuple = tuple(sorted(params.items()))
            print(f"Target: {target_name} \n Params: {params_tuple}, Param_tuple: {params_tuple} \n")
            if params_tuple not in saved_configs:
                # This configuration is new, create a new folder
                folder_name = "_".join([t for t in TARGET_VARIABLES_WITH_MULTI
                                        if tuple(sorted(self.best_params[t].items())) == params_tuple]) + "_model"
                saved_configs[params_tuple] = folder_name  # Store the folder name
                model_folder = os.path.join(self.save_dir, folder_name)
                os.makedirs(model_folder, exist_ok=True)
                with open(os.path.join(model_folder, "params.json"), "w") as f:
                    json.dump(params, f)
        return saved_configs

    def save_model(self, model, filename):
        joblib.dump(model, os.path.join(self.save_dir, filename))

    def save_best_params(self, filename):
        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump(self.best_params, f)

    def plot_chosen_configurations_rmse(self, best_rmses):
        """Bar plot of RMSE scores for the chosen configuration."""
        labels = TARGET_VARIABLES_WITH_MULTI
        rmse_values = [best_rmses[target] for target in labels]

        plt.figure(figsize=(12, 6))
        colors = [COLOR_PALETTE.get(target, '#D3D3D3')[0] for target in labels]  # Use the first color for each target
        valid_colors = [color if color != '#' else '#D3D3D3' for color in colors]
        bars = plt.bar(labels, rmse_values, color=valid_colors)

        # Add RMSE scores on top of each bar
        for bar, rmse in zip(bars, rmse_values):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(rmse, 2), va='bottom', ha='center')

        plt.title("RMSE Scores for Chosen Configurations")
        plt.xlabel("Target Variable")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_figure_dir, "chosen_configurations_rmse.png"))
        plt.show()

    def find_best_configuration(self, X_train, y_train, X_val, y_val):
        # find the best hyperparameters for each target variable and the mean
        param_grid = self.get_param_grid()

        # Initialize the best RMSE and parameters storage
        best_rmses = {target: float("inf") for target in TARGET_VARIABLES_WITH_MULTI}
        best_params = {target: None for target in TARGET_VARIABLES_WITH_MULTI}
        best_models = {target: None for target in TARGET_VARIABLES_WITH_MULTI}

        # Hyperparameter tuning loop
        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            # todo: delete mean_rmse
            rmses, mean_rmse, models, multi_model, multi_rmse = self.train_and_evaluate(params, X_train, y_train,
                                                                                        X_val, y_val)
            # Update best RMSEs and models for each target and mean
            for i, target in enumerate(TARGET_VARIABLES):
                if rmses[i] < best_rmses[target]:
                    best_rmses[target] = rmses[i]
                    best_params[target] = params
                    best_models[target] = models[i]  # Store the individual model

            # Update for multi-output model
            if multi_rmse < best_rmses[MUTLI]:
                best_rmses[MUTLI] = multi_rmse
                best_params[MUTLI] = params
                best_models[MUTLI] = multi_model  # Store the multi-output model

        self.best_params = best_params
        self.model = best_models  # Store the best models dictionary
        self.plot_chosen_configurations_rmse(best_rmses)

        saved_config_folders = self.save_configurations()

        return best_rmses, best_params, saved_config_folders

    def cross_validate_and_evaluate(self, X, y, params, config_folder, test_path):
        """Performs cross-validation, evaluates on validation and test sets, and saves the model."""
        folder_name_for_print = os.path.basename(config_folder)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        cv_rmses = []

        print(f"Cross-validating {folder_name_for_print}...")
        # Initialize dictionaries to store RMSEs for each fold
        fold_train_rmses = {key: [] for key in TARGET_VARIABLES_WITH_MULTI}
        fold_val_rmses = {key: [] for key in TARGET_VARIABLES_WITH_MULTI}

        # Create and train individual models for each target
        for train_index, val_index in tqdm(kf.split(X), desc="Cross-validation", disable=False):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            # Create and train individual models for each target
            models = []
            fold_rmses = []
            for i, target in enumerate(y_train_fold.columns):
                eval_set = [(X_train_fold, y_train_fold.iloc[:, i]), (X_val_fold, y_val_fold.iloc[:, i])]
                model_params = {k: v for k, v in params.items() if k not in ['eval_set', 'eval_metric', 'verbose']}
                model = xgb.XGBRegressor(**model_params, eval_metric='rmse')
                model.fit(X_train_fold, y_train_fold.iloc[:, i], eval_set=eval_set, verbose=False)
                models.append(model)

                y_pred_val_fold = model.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold.iloc[:, i], y_pred_val_fold))
                fold_rmses.append(rmse)

                # Store the RMSEs for each fold
                fold_train_rmses[target].append(model.evals_result()["validation_0"]["rmse"])
                fold_val_rmses[target].append(model.evals_result()["validation_1"]["rmse"])

            # Multi-output model
            model_params = {k: v for k, v in params.items() if k not in ['eval_set', 'eval_metric', 'verbose']}
            multi_model = MultiOutputRegressor(xgb.XGBRegressor(**model_params, eval_metric='rmse'))
            multi_model.fit(X_train_fold, y_train_fold)
            y_pred_val_fold_multi = multi_model.predict(X_val_fold)
            rmse_multi = np.sqrt(mean_squared_error(y_val_fold, y_pred_val_fold_multi, multioutput='uniform_average'))

            fold_train_rmses[MUTLI].append(rmse_multi)
            fold_val_rmses[MUTLI].append(rmse_multi)
            cv_rmses.append(np.mean(fold_rmses))  # Append the mean RMSE across targets for the fold

        # Store the RMSEs for each target
        self.train_rmses = {key: np.mean(fold_train_rmses[key], axis=0) for key in TARGET_VARIABLES}
        self.val_rmses = {key: np.mean(fold_val_rmses[key], axis=0) for key in TARGET_VARIABLES}

        # Calculate the mean RMSE across the target variables
        self.train_rmses['mean'] = np.mean([self.train_rmses[key] for key in TARGET_VARIABLES], axis=0)
        self.val_rmses['mean'] = np.mean([self.val_rmses[key] for key in TARGET_VARIABLES], axis=0)

        self.train_rmses[MUTLI] = np.mean(fold_train_rmses[MUTLI], axis=0)
        self.val_rmses[MUTLI] = np.mean(fold_val_rmses[MUTLI], axis=0)

        mean_cv_rmse = np.mean(cv_rmses)
        print(f"Mean CV RMSE for {folder_name_for_print}: {mean_cv_rmse}")
        print(f'Mean CV RMSE for Multi: {self.val_rmses[MUTLI]}')

        # todo: I would like to evaluate all the models that trained for a specific target variable,
        #  to predict over all the data on the validation set and test set

        # Evaluate on the full validation set
        print(f"Evaluating model for targets: {folder_name_for_print} on validation set...")
        y_pred_val = np.zeros((len(self.val_data), len(TARGET_VARIABLES)))

        for i, target in enumerate(TARGET_VARIABLES):
            y_pred_val[:, i] = models[i].predict(self.val_data.drop(NON_FEATURE_COLUMNS, axis=1))
        val_rmse = np.sqrt(
            mean_squared_error(self.val_data[TARGET_VARIABLES], y_pred_val, multioutput="uniform_average"))
        print(f"Validation RMSE for targets {folder_name_for_print}: {val_rmse:.4f}")

        # Multi-output validation evaluation
        y_pred_val_multi = multi_model.predict(self.val_data.drop(NON_FEATURE_COLUMNS, axis=1))
        val_rmse_multi = np.sqrt(
            mean_squared_error(self.val_data[TARGET_VARIABLES], y_pred_val_multi, multioutput="uniform_average"))
        print(f"Validation RMSE for multi-output model: {val_rmse_multi:.4f}")

        # Evaluate on the test set
        print(f"Evaluating model for targets: {folder_name_for_print} on test set...")
        test_data = pd.read_parquet(test_path)
        X_test = test_data.drop(NON_FEATURE_COLUMNS, axis=1)
        y_test = test_data[TARGET_VARIABLES]
        y_pred_test = np.zeros((len(test_data), len(TARGET_VARIABLES)))

        for i, target in enumerate(TARGET_VARIABLES):
            y_pred_test[:, i] = models[i].predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput="uniform_average"))
        print(f"Test RMSE for targets {folder_name_for_print}: {test_rmse:.4f}")

        # Multi-output test evaluation
        y_pred_test_multi = multi_model.predict(X_test)
        test_rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_test_multi, multioutput="uniform_average"))
        print(f"Test RMSE for multi-output model: {test_rmse_multi:.4f}")

        # Save the model in the config folder
        for i, target in enumerate(TARGET_VARIABLES):
            self.save_model(models[i], os.path.join(folder_name_for_print, "model.pkl"))

        # Save the multi-output model
        self.save_model(multi_model, os.path.join(folder_name_for_print, "multi_model.pkl"))

        return mean_cv_rmse, val_rmse, test_rmse, val_rmse_multi, test_rmse_multi

    def run(self, train_path, val_path, test_path):
        self.load_data(train_path, val_path)
        X_train, y_train = self.preprocess_data(dataset="train")
        X_val, y_val = self.preprocess_data(dataset="val")

        print("Finding best configurations...")
        best_rmses, best_params, saved_config_folders = self.find_best_configuration(X_train, y_train, X_val, y_val)
        print("Best RMSEs:")
        for key, value in best_rmses.items():
            print(f"{key}: {value}")

        print("\nBest Params:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

        results = {}
        for config_folder in saved_config_folders.values():
            full_path = os.path.join(self.save_dir, config_folder)
            cv_rmse, val_rmse, test_rmse, val_rmse_multi, test_rmse_multi = self.cross_validate_and_evaluate(
                X_train, y_train, best_params, full_path, test_path)
            results[config_folder] = {"cv_rmse": cv_rmse, "val_rmse": val_rmse, "test_rmse": test_rmse,
                                      "val_rmse_multi": val_rmse_multi, "test_rmse_multi": test_rmse_multi}

        print("\nSummary of Results:")
        for config_name, metrics in results.items():
            print(f"Model: {config_name}")
            print(f"  CV RMSE: {metrics['cv_rmse']:.4f}")
            print(f"  Validation RMSE: {metrics['val_rmse']:.4f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"  Validation RMSE (Multi): {metrics['val_rmse_multi']:.4f}")
            print(f"  Test RMSE (Multi): {metrics['test_rmse_multi']:.4f}")

        # Choose the best model based on a specific metric (e.g., validation RMSE)
        best_model_config = min(results, key=lambda k: (results[k]['test_rmse'], results[k]['test_rmse_multi']))
        print(f"\nBest Model based on Validation RMSE: {best_model_config}")
