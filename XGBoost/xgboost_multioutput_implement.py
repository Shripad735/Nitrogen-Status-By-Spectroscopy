import os
import json
import shutil
import joblib
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from constants_config import TARGET_VARIABLES, NON_FEATURE_COLUMNS, COLOR_PALETTE, MULTI, TARGET_VARIABLES_WITH_MULTI, \
    TARGET_VARIABLES_WITH_MULTIS, MODELS, TARGET_VARIABLES_WITH_MEAN, MEAN


class XGBoostMultiOutput:
    def __init__(self, n_splits=2, save_dir='models/', save_figure_dir='figures/'):
        self.data = None
        # self.train_data = None
        # self.train_data_plsr = None
        # self.val_data = None
        # self.val_data_plsr = None
        # self.test_data = None
        # self.test_data_plsr = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_figure_dir = save_figure_dir
        os.makedirs(self.save_figure_dir, exist_ok=True)
        self.model = None
        self.best_params = None
        self.train_rmses = {}
        self.val_rmses = {}

    def load_data(self, train_path, val_path, test_path):
        self.train_data = pd.read_parquet(train_path)
        self.val_data = pd.read_parquet(val_path)
        self.test_data = pd.read_parquet(test_path)

    def preprocess_data(self, dataset="train"):
        feature_columns = [col for col in self.train_data.columns if col not in NON_FEATURE_COLUMNS]
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[TARGET_VARIABLES]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[TARGET_VARIABLES]
        elif dataset == "test":
            X = self.test_data[feature_columns]
            y = self.test_data[TARGET_VARIABLES]
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

    def train_and_evaluate_by_rmse_per_configuration(self, params, X_train, y_train, X_val, y_val):
        # Multi-output regressor model
        model_params = {k: v for k, v in params.items() if k not in ['eval_set', 'eval_metric', 'verbose']}
        model = MultiOutputRegressor(xgb.XGBRegressor(**model_params, eval_metric='rmse'))
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        # Calculate RMSE for each target variable of the multi_model
        multi_rmses = np.sqrt(mean_squared_error(y_val, y_pred_val, multioutput='raw_values'))

        return model, np.mean(multi_rmses), multi_rmses.tolist()

    def save_configurations(self, model_name, best_params):
        """Save the best configurations (parameters) and their corresponding models."""
        print("Saving best configurations...")
        # saved_configs = {}
        print("\nBest Params:")
        # # Track saved configurations and their folder names
        # for target_name, params in self.best_params.items():
        # Create a unique folder name for the parameters
        # params_tuple = tuple(sorted(params.items()))
        # print(f"{target_name}: {params}")
        folder_name = f"{model_name}_model"
        # saved_configs[params_tuple] = folder_name  # Store the folder name
        model_folder = os.path.join(self.save_dir, folder_name)
        os.makedirs(model_folder, exist_ok=True)
        with open(os.path.join(model_folder, "params.json"), "w") as f:
            json.dump(best_params, f)

    def save_model(self, model, target_name, filename):
        joblib.dump(model, os.path.join(self.save_dir, target_name, filename))

    def save_best_params(self, filename):
        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump(self.best_params, f)

    def plot_chosen_configurations_rmse(self, best_targets_rmses, best_multi_rmse):
        """Bar plot of RMSE scores for the chosen configuration."""
        labels = TARGET_VARIABLES_WITH_MEAN
        rmse_values = [best_targets_rmses[target] for target in TARGET_VARIABLES] + [best_multi_rmse]

        plt.figure(figsize=(12, 6))
        colors = [COLOR_PALETTE.get(target, '#D3D3D3')[0] for target in TARGET_VARIABLES] + [
            COLOR_PALETTE.get(MEAN, '#D3D3D3')[0]]

        # Validate and convert colors
        valid_colors = []
        for color in colors:
            try:
                valid_colors.append(mcolors.to_rgba(color))
            except (ValueError, TypeError):
                valid_colors.append(mcolors.to_rgba('#D3D3D3'))

        bars = plt.bar(labels, rmse_values, color=valid_colors)

        # Add RMSE scores on top of each bar
        for bar, rmse in zip(bars, rmse_values):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(rmse, 2), va='bottom', ha='center')

        # Manually add legend entries for each target variable, including 'multi'
        legend_handles = []
        for target, color in zip(labels, valid_colors):
            individual_patch = plt.Line2D([0], [0], color=color, lw=4, label=f'{target} RMSE')
            legend_handles.append(individual_patch)

        plt.legend(handles=legend_handles)

        plt.title("RMSE Scores for Chosen Configurations")
        plt.xlabel("Target Variable")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_figure_dir, "chosen_configurations_rmse.png"))
        plt.show()

    def find_best_configuration_based_rmse_score(self, X_train, y_train, X_val, y_val, model_name):
        # find the best hyperparameters for each target variable and the mean
        param_grid = self.get_param_grid()

        # Initialize the best RMSE and parameters storage
        minimal_rmse = float("inf")
        best_params = {}
        model = {}
        best_targets_rmses = {target: None for target in TARGET_VARIABLES}

        # Hyperparameter tuning loop
        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            evaluated_model, rmse, multi_targets_rmses = self.train_and_evaluate_by_rmse_per_configuration(
                params, X_train, y_train, X_val, y_val)

            # Update for multi-output model
            if rmse < minimal_rmse:
                minimal_rmse = rmse
                best_params = params
                model = evaluated_model  # Store the multi-output model
                best_targets_rmses = {target: multi_targets_rmses[i] for i, target in enumerate(TARGET_VARIABLES)}

        self.best_params = best_params
        self.model = model  # Store the best models dictionary
        self.plot_chosen_configurations_rmse(best_targets_rmses, minimal_rmse)
        self.save_configurations(model_name, best_params)

        return best_targets_rmses, best_params, minimal_rmse

    def k_fold_cross_validate_model(self, X_train, y_train, model_name, params):
        """Performs k-fold cross-validation and evaluation on the validation and test sets."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Initialize dictionaries to store RMSEs for each fold for each of n_estimators
        fold_train_rmses = {key: [] for key in TARGET_VARIABLES}
        fold_val_rmses = {key: [] for key in TARGET_VARIABLES}

        # Train multi-output model
        multi_model_params = params[MULTI]
        for train_index, val_index in tqdm(kf.split(X_train), desc="Cross-validation for Multi", disable=False):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = MultiOutputRegressor(
                xgb.XGBRegressor(**multi_model_params, eval_metric='rmse')
            )

            # Fit the multi-output model
            model.fit(X_train_fold, y_train_fold)

            # Train each estimator with eval_set
            for i, estimator in enumerate(model.estimators_):
                eval_set = [(X_train_fold, y_train_fold.iloc[:, i]), (X_val_fold, y_val_fold.iloc[:, i])]
                estimator.fit(X_train_fold, y_train_fold.iloc[:, i], eval_set=eval_set, verbose=False)

            # Store the RMSEs for each fold
            for i, target in enumerate(y_train.columns):
                fold_train_rmses[f'{target}'].append(
                    model.estimators_[i].evals_result()["validation_0"]["rmse"]
                )
                fold_val_rmses[f'{target}'].append(
                    model.estimators_[i].evals_result()["validation_1"]["rmse"]
                )

        # Save the multi-output model
        # todo - No such file or directory: 'models/model name/model name.pkl'
        self.save_model(model, model_name, os.path.join(f"{model_name}.pkl"))

        # Store the RMSEs for each target
        self.train_rmses = {f'{key}_mean_folds': np.mean(fold_train_rmses[key], axis=0) for key in TARGET_VARIABLES}
        self.val_rmses = {f'{key}_mean_folds': np.mean(fold_val_rmses[key], axis=0) for key in TARGET_VARIABLES}

    def evaluate_models_on_validation_sets(self, target, X_val, y_val, model):
        """Evaluates the model on the validation set."""
        if target != MULTI:
            y_pred_val = model.predict(X_val)
            val_rmse_individual = np.sqrt(mean_squared_error(y_val[target], y_pred_val))
            self.val_rmses[target] = val_rmse_individual
            print(f"Validation RMSE for {target}: {val_rmse_individual:.4f}")
        else:
            y_pred_val_multi = model.predict(X_val)
            val_rmse_multi_individuals = np.sqrt(mean_squared_error(y_val, y_pred_val_multi, multioutput='raw_values'))
            for i, target in enumerate(TARGET_VARIABLES):
                self.val_rmses[f'{MULTI}_{target}'] = val_rmse_multi_individuals[i]
                print(f"Validation RMSE for multi-output model {target}: {val_rmse_multi_individuals[i]:.4f}")

            val_rmse_multi = np.mean(np.sqrt(mean_squared_error(y_val, y_pred_val_multi, multioutput='raw_values')))
            self.val_rmses[MULTI] = val_rmse_multi
            print(f"Validation RMSE for multi-output model (average): {val_rmse_multi:.4f}")

    def get_best_model(self) -> str:
        """ get the mean RMSE of all the target variables and compare the mean to the mean RMSE of the multi-output
        model, if the lower RMSE is the multi-output one than save it in as the best model, else save all 3
        individuals models in XGBoost_final_model folder"""
        best_model = ''
        mean_individual_rmses = np.mean([self.val_rmses[target] for target in TARGET_VARIABLES])
        print(f"Mean RMSE for individual models: {mean_individual_rmses:.4f}")
        print(f"Mean RMSE for multi-output model: {self.val_rmses[MULTI]:.4f}")
        os.makedirs('XGBoost_final_model', exist_ok=True)

        if self.val_rmses[MULTI] <= mean_individual_rmses:
            best_model = MULTI
            self.model = self.model[best_model]
            self.best_params = self.best_params[best_model]
            print("Individual models have higher average RMSE. Saving the multi-output model.")
        else:
            best_model = TARGET_VARIABLES
            print("Multi-output model has higher average RMSE. Saving the individual models.")
            self.model = {target: self.model[target] for target in best_model}
            self.best_params = {target: self.best_params[target] for target in best_model}
        return best_model

    def evaluate_best_model_on_test_set(self, test_path, best_model):
        """ get the model from the best model and evaluate it on the test set """
        test_data = pd.read_parquet(test_path)
        X_test = test_data.drop(NON_FEATURE_COLUMNS, axis=1)
        y_test = test_data[TARGET_VARIABLES]
        y_pred_test = np.zeros((len(test_data), len(TARGET_VARIABLES)))

        individual_rmses = []
        if best_model == MULTI:
            y_pred_test = self.model.predict(X_test)
            test_rmse_multi_individuals = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput='raw_values'))
            for i, target in enumerate(TARGET_VARIABLES):
                print(f"Test RMSE for multi-output model {target}: {test_rmse_multi_individuals[i]:.4f}")
            test_rmse_multi = np.mean(np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput='raw_values')))
            print(f"Test RMSE for multi-output model (average): {test_rmse_multi:.4f}")
        else:
            for i, target in enumerate(TARGET_VARIABLES):
                y_pred_test[:, i] = self.model[target].predict(X_test)
                test_rmse_individual = np.sqrt(mean_squared_error(y_test[target], y_pred_test[:, i]))
                individual_rmses.append(test_rmse_individual)
                print(f"Test RMSE for {target}: {test_rmse_individual:.4f}")

            # Calculate the average RMSE for the individual models
            avg_test_rmse_individual = np.mean(individual_rmses)
            print(f"Average Test RMSE for individual models: {avg_test_rmse_individual:.4f}")

    def run(self, train_path, val_path, test_path, model_name):
        # Load the data
        self.load_data(train_path, val_path, test_path)
        X_train, y_train = self.preprocess_data(dataset="train")
        X_val, y_val = self.preprocess_data(dataset="val")
        X_test, y_test = self.preprocess_data(dataset="test")

        print("Finding best configurations...")
        best_rmses, best_params, best_multi_rmses = self.find_best_configuration_based_rmse_score(
            X_train, y_train, X_val, y_val, model_name)
        print("\nBest RMSEs:")
        for key, value in best_rmses.items():
            print(f"{key}: {value}")

        print(f"{MEAN}: {best_multi_rmses}")

        # todo - continue from here the
        # for each variable do the K-fold cross-validate and evaluate on the validation and test sets
        params = {}
        for target in TARGET_VARIABLES_WITH_MULTI:
            folder_name = f"{target}_model"
            params_path = os.path.join(self.save_dir, folder_name, "params.json")
            with open(params_path, "r") as f:
                params[target] = json.load(f)

        # todo - ready for one model - multi / plsr
        self.k_fold_cross_validate_model(X_train, y_train, 'model name', params)

        for target in TARGET_VARIABLES_WITH_MULTI:
            model = joblib.load(os.path.join(self.save_dir, f"{target}_model", f"{target}_model.pkl"))
            self.evaluate_models_on_validation_sets(target, X_val, y_val, model)

        best_model = self.get_best_model()
        print(f"Best Model: {best_model}")

        self.evaluate_best_model_on_test_set(test_path, best_model)
