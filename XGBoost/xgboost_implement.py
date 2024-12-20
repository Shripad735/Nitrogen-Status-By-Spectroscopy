import os
import shap
import json
import joblib
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from pdpbox import pdp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from constants_config import TARGET_VARIABLES, NON_FEATURE_COLUMNS


class XGBoostMultiOutput:
    def __init__(self, n_splits=2, save_dir="models/"):
        self.val_data = None
        self.train_data = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model = None
        self.best_params = None

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
        # Create separate eval_sets for each target variable
        eval_sets = [[(X_train, y_train.iloc[:, i]), (X_val, y_val.iloc[:, i])] for i in range(y_train.shape[1])]

        model = MultiOutputRegressor(xgb.XGBRegressor(**params, eval_sets=eval_sets, eval_metric='rmse', verbose=False))

        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        rmses = np.sqrt(mean_squared_error(y_val, y_pred_val, multioutput="raw_values"))
        mean_rmse = np.mean(rmses)
        return rmses, mean_rmse, model  # Return individual and mean RMSEs

    def save_configurations(self):
        # Save the best configurations (parameters) and their corresponding target names
        print("Saving best configurations...")
        saved_configs = {}  # To keep track of saved configurations and their folder names
        for target_name, params in self.best_params.items():
            params_tuple = tuple(sorted(params.items()))
            print(f"Target: {target_name}, Params: {params_tuple}, Param_tuple: {params_tuple} \n")
            if params_tuple not in saved_configs:
                # This configuration is new, create a new folder
                folder_name = "_".join([t for t in TARGET_VARIABLES + ["mean"]
                                        if tuple(sorted(self.best_params[t].items())) == params_tuple]) + "_model"
                saved_configs[params_tuple] = folder_name  # Store the folder name
                model_folder = os.path.join(self.save_dir, folder_name)
                os.makedirs(model_folder, exist_ok=True)
                with open(os.path.join(model_folder, "params.json"), "w") as f:
                    json.dump(params, f)
        return saved_configs

    def plot_chosen_configurations_rmse(self, best_rmses):
        """Bar plot of RMSE scores for the chosen configuration."""
        labels = TARGET_VARIABLES + ["mean"]
        rmse_values = [best_rmses[target] for target in labels]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, rmse_values, color=['#FFB6C1', '#B0E0E6', '#FFDAB9', '#D3D3D3'])
        # Add RMSE scores on top of each bar
        for bar, rmse in zip(bars, rmse_values):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(rmse, 2), va='bottom', ha='center')

        plt.title("RMSE Scores for Chosen Configurations")
        plt.xlabel("Target Variable")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "chosen_configurations_rmse.png"))
        plt.show()

    def find_best_configuration(self, X_train, y_train, X_val, y_val):
        param_grid = self.get_param_grid()
        best_rmses = {target: float("inf") for target in TARGET_VARIABLES + ["mean"]}  # store best RMSEs
        best_params = {target: None for target in TARGET_VARIABLES + ["mean"]}  # store best params follows RMSEs
        best_models = {target: None for target in TARGET_VARIABLES + ["mean"]}  # store best models follows RMSEs

        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            rmses, mean_rmse, model = self.train_and_evaluate(params, X_train, y_train, X_val, y_val)

            # Update best RMSEs and models for each target and mean
            for i, target in enumerate(TARGET_VARIABLES):
                if rmses[i] < best_rmses[target]:
                    best_rmses[target] = rmses[i]
                    best_params[target] = params
                    best_models[target] = model.estimators_[i]  # Save individual estimator
            if mean_rmse < best_rmses["mean"]:
                best_rmses["mean"] = mean_rmse
                best_params["mean"] = params
                best_models["mean"] = model  # Save the entire MultiOutputRegressor

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
        for train_index, val_index in tqdm(kf.split(X), desc="Cross-validation"):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            eval_sets = [[(X_train_fold, y_train_fold.iloc[:, i]), (X_val_fold, y_val_fold.iloc[:, i])]
                         for i in range(y_train_fold.shape[1])]

            # Initialize a NEW model for each fold
            model = MultiOutputRegressor(xgb.XGBRegressor(**params, eval_set=eval_sets, eval_metric='rmse',
                                                          verbose=False))

            model.fit(X_train_fold, y_train_fold)
            # Evaluate on the validation fold
            y_pred_val_fold = model.predict(X_val_fold)
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val_fold, multioutput="raw_values")).mean()
            cv_rmses.append(fold_rmse)

        mean_cv_rmse = np.mean(cv_rmses)
        print(f"Mean CV RMSE for {folder_name_for_print}: {mean_cv_rmse}")

        # Evaluate on the full validation set
        print(f"Evaluating model for targets: {folder_name_for_print} on validation set...")
        y_pred_val = model.predict(self.val_data.drop(NON_FEATURE_COLUMNS, axis=1))
        val_rmse = np.sqrt(
            mean_squared_error(self.val_data[TARGET_VARIABLES], y_pred_val, multioutput="raw_values")).mean()
        print(f"Validation RMSE for targets {folder_name_for_print}: {val_rmse:.4f}")

        # Evaluate on the test set
        print(f"Evaluating model for targets: {folder_name_for_print} on test set...")
        test_data = pd.read_parquet(test_path)
        X_test = test_data.drop(NON_FEATURE_COLUMNS, axis=1)
        y_test = test_data[TARGET_VARIABLES]
        y_pred_test = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput="raw_values")).mean()
        print(f"Test RMSE for targets {folder_name_for_print}: {test_rmse:.4f}")

        # Save the model in the config folder
        self.save_model(model, os.path.join(folder_name_for_print, "model.pkl"))

        return mean_cv_rmse, val_rmse, test_rmse

    def plot_learning_curve(self, model):
        # Take the average learning curve across all outputs
        n_estimators = model.estimators_[0].n_estimators  # Assuming all estimators have the same n_estimators
        x_axis = range(1, n_estimators + 1)

        plt.figure(figsize=(10, 6))

        train_rmses = []
        val_rmses = []
        for i in range(len(model.estimators_)):
            train_rmses.append(model.estimators_[i].evals_result()["validation_0"]["rmse"])
            val_rmses.append(model.estimators_[i].evals_result()["validation_1"]["rmse"])

        plt.plot(x_axis, np.mean(train_rmses, axis=0), label="Average Train RMSE")
        plt.plot(x_axis, np.mean(val_rmses, axis=0), label="Average Validation RMSE")

        plt.title("Average Learning Curve")
        plt.xlabel("Boosting Round")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, "average_learning_curve.png"))
        plt.show()

    def plot_feature_importances(self, model, X_train):
        """Plots mean feature importance."""
        all_importances = [
            est.feature_importances_ for est in model.estimators_
        ]
        mean_importance = np.mean(all_importances, axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1]
        feature_names = X_train.columns[sorted_idx]
        mean_importance = mean_importance[sorted_idx]

        plt.figure(figsize=(12, 6))
        plt.bar(feature_names[:20], mean_importance[:20])
        plt.xticks(rotation=90)
        plt.title("Mean Feature Importance")
        plt.xlabel("Feature")
        plt.ylabel("Mean Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "mean_feature_importance.png"))
        plt.show()

    def save_model(self, model, filename):
        joblib.dump(model, os.path.join(self.save_dir, filename))

    def save_best_params(self, filename):
        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump(self.best_params, f)

    def plot_residuals(self, model, X, y):
        """Plots residuals for all target variables combined."""
        y_pred = model.predict(X)
        residuals = y - y_pred
        residuals = residuals.values.flatten()  # Flatten to a 1D array

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred.flatten(), residuals, alpha=0.5)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Combined Residuals Plot")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "combined_residuals_plot.png"))
        plt.show()

    def plot_predicted_vs_actual(self, model, X, y):
        """Plots predicted vs actual values for all target variables combined."""
        y_pred = model.predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(y.values.flatten(), y_pred.flatten(), alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Combined Predicted vs. Actual Plot")
        plt.plot(
            [y.min().min(), y.max().max()],
            [y.min().min(), y.max().max()],
            "k--", lw=2)  # Add a diagonal line for reference
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "combined_predicted_vs_actual.png"))
        plt.show()

    def plot_rmse_per_fold(self, rmses):
        """Plots RMSE scores for each cross-validation fold."""
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(rmses) + 1), rmses)
        plt.xlabel("Fold")
        plt.ylabel("RMSE")
        plt.title("RMSE per Cross-Validation Fold")
        plt.savefig(os.path.join(self.save_dir, "rmse_per_fold.png"))
        plt.show()

    def plot_pdp(self, model, X, feature):
        """Plots a Partial Dependence Plot for a single feature using the first estimator."""
        pdp_feature = pdp.pdp_isolate(
            model=model.estimators_[0],
            dataset=X,
            model_features=X.columns,
            feature=feature,
        )
        pdp.pdp_plot(pdp_isolate_out=pdp_feature, feature_name=feature)
        plt.savefig(os.path.join(self.save_dir, f"pdp_{feature}.png"))
        plt.show()

    def plot_shap_summary(self, model, X):
        """Plots a SHAP summary plot using the first estimator."""
        explainer = shap.TreeExplainer(model.estimators_[0])
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)
        plt.savefig(os.path.join(self.save_dir, "shap_summary_plot.png"))

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
            cv_rmse, val_rmse, test_rmse = self.cross_validate_and_evaluate(X_train, y_train, best_params, full_path,
                                                                            test_path)
            results[config_folder] = {"cv_rmse": cv_rmse, "val_rmse": val_rmse, "test_rmse": test_rmse}

        print("\nSummary of Results:")
        for config_name, metrics in results.items():
            print(f"Model: {config_name}")
            print(f"  CV RMSE: {metrics['cv_rmse']:.4f}")
            print(f"  Validation RMSE: {metrics['val_rmse']:.4f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.4f}")

        # Choose the best model based on a specific metric (e.g., validation RMSE)
        best_model_config = min(results, key=lambda k: results[k]['test_rmse'])
        print(f"\nBest Model based on Validation RMSE: {best_model_config}")

        # Load the best model's parameter for further analysis or deployment
        best_params_path = os.path.join(self.save_dir, best_model_config, "params.json")
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)

        # Retrain the best model on the FULL training data
        eval_sets = [[(X_train, y_train.iloc[:, i]), (X_val, y_val.iloc[:, i])] for i in range(y_train.shape[1])]
        # Create a new MultiOutputRegressor with the best parameters
        best_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params, eval_sets=eval_sets, eval_metric='rmse',
                                                           verbose=False))
        best_model.fit(X_train, y_train)

        # Additional Visualizations
        self.plot_learning_curve(best_model)
        self.plot_feature_importances(best_model, X_train)
        self.plot_residuals(best_model, X_val, y_val)
        self.plot_predicted_vs_actual(best_model, X_val, y_val)

        # Save the model
        self.save_model(best_model, "XGBoost_final_model/model.pkl")
        self.save_best_params("XGBoost_final_model/params.json")

        # Advanced Visualizations (Use the first estimator as an example)
        try:
            self.plot_pdp(best_model, X_val, feature="your_feature_name")
            self.plot_shap_summary(best_model, X_val)
        except NameError:
            print("Skipping advanced visualizations (PDP and SHAP). Install pdpbox and shap for these plots.")