import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants_config import TARGET_VARIABLES_WITH_MULTI

from constants_config import COLOR_PALETTE, TARGET_VARIABLES_WITH_MEAN, NON_FEATURE_COLUMNS, TARGET_VARIABLES


def save_best_model_and_params(model, params, train_rmses, val_rmses, directory="XGBoost_final_model"):
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, os.path.join(directory, "model.pkl"))
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f)

    train_rmses_to_save = {key: train_rmses[key] for key in
                           ['N_Value_mean_folds', 'SC_Value_mean_folds', 'ST_Value_mean_folds']}
    val_rmses_to_save = {key: val_rmses[key] for key in
                         ['N_Value_mean_folds', 'SC_Value_mean_folds', 'ST_Value_mean_folds']}
    final_train_rmses = {'N_Value': train_rmses_to_save['N_Value_mean_folds'],
                         'SC_Value': train_rmses_to_save['SC_Value_mean_folds'],
                         'ST_Value': train_rmses_to_save['ST_Value_mean_folds']}
    final_val_rmses = {'N_Value': val_rmses_to_save['N_Value_mean_folds'],
                       'SC_Value': val_rmses_to_save['SC_Value_mean_folds'],
                       'ST_Value': val_rmses_to_save['ST_Value_mean_folds']}
    with open(os.path.join(directory, "train_rmses.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in final_train_rmses.items()}, f)
    with open(os.path.join(directory, "val_rmses.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in final_val_rmses.items()}, f)


# todo: continue from here - validate it works
def load_best_model_and_params(directory="XGBoost_final_model"):
    models = {}
    for target in TARGET_VARIABLES_WITH_MULTI:
        model_path = os.path.join(directory, f"{target}_model.pkl")
        if os.path.exists(model_path):
            models[target] = joblib.load(model_path)

    multi_model_path = os.path.join(directory, "multi_model.pkl")
    if os.path.exists(multi_model_path):
        models['Multi'] = joblib.load(multi_model_path)

    model = joblib.load(os.path.join(directory, "model.pkl"))
    with open(os.path.join(directory, "params.json"), "r") as f:
        params = json.load(f)
    with open(os.path.join(directory, "train_rmses.json"), "r") as f:
        train_rmses = json.load(f)
    with open(os.path.join(directory, "val_rmses.json"), "r") as f:
        val_rmses = json.load(f)
    return model, params, train_rmses, val_rmses


def get_X_y(data_path, dataset='train'):
    data = pd.read_parquet(data_path)
    feature_columns = [col for col in data.columns if col not in NON_FEATURE_COLUMNS]
    if dataset in ('train', 'val', 'test'):
        X = data[feature_columns]
        y = data[TARGET_VARIABLES]
    return X, y


def plot_learning_curve(xgb_multi_output, config_name, save_dir):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    minimal_individual_estimators = float("inf")
    for var in TARGET_VARIABLES:
        n_estimators = len(xgb_multi_output.train_rmses[f'{var}_mean_folds'])
        if n_estimators < minimal_individual_estimators:
            minimal_individual_estimators = n_estimators

    n_estimators = min(xgb_multi_output.best_params['n_estimators'], minimal_individual_estimators)
    x_axis = range(1, n_estimators + 1)

    for var in TARGET_VARIABLES:

        train_rmse_mean = xgb_multi_output.train_rmses[f'{var}_mean_folds'][:n_estimators]
        val_rmse_mean = xgb_multi_output.val_rmses[f'{var}_mean_folds'][:n_estimators]

        train_color, val_color = COLOR_PALETTE.get(var, ('#D3D3D3', '#DCDCDC'))  # default to light gray

        plt.plot(x_axis, train_rmse_mean, label=f"Train RMSE - {var}", color=train_color)
        plt.plot(x_axis, val_rmse_mean, label=f"Validation RMSE - {var}", color=val_color)

    plt.title(f"All Learning Curves Based On The Best Configuration")
    plt.xlabel("Number of Estimators")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"learning_curve_{config_name}.png"))
    plt.show()


def plot_feature_importances(xgb_multi_output, save_dir):
    """Plots mean feature importance."""
    all_importances = []
    for key, model in xgb_multi_output.model.items():
        if isinstance(model, list):
            for est in model:
                all_importances.append(est.feature_importances_)
        else:
            all_importances.append(model.feature_importances_)

    mean_importance = np.mean(all_importances, axis=0)
    sorted_idx = np.argsort(mean_importance)[::-1]
    X_train, _ = get_X_y('../datasets/train_data.parquet', dataset='train')
    feature_names = X_train.columns[sorted_idx]
    mean_importance = mean_importance[sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.bar(feature_names[:20], mean_importance[:20])
    plt.xticks(rotation=90)
    plt.title("Mean Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Mean Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mean_feature_importance.png"))
    plt.show()


def plot_residuals(xgb_multi_output, save_dir):
    X_val, y_val = get_X_y('../datasets/test_data.parquet', dataset='val')
    y_pred = np.zeros((X_val.shape[0], len(xgb_multi_output.model)))

    for var in xgb_multi_output.model:
        if isinstance(xgb_multi_output.model[var], list):
            pass
        else:
            y_pred[:, list(xgb_multi_output.model.keys()).index(var)] = xgb_multi_output.model[var].predict(X_val)

    # Slice the first three columns from y_val and y_pred
    y_val_first_three = y_val.iloc[:, :3]
    y_pred_first_three = y_pred[:, :3]

    # Calculate residuals for the first three columns
    residuals = y_val_first_three.values - y_pred_first_three
    residuals = residuals.flatten()  # Flatten to a 1D array

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_first_three.flatten(), residuals, alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals Values")
    plt.title("Residuals Plot for For Target Variables")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "residuals_plot_for_target_values.png"))
    plt.show()
