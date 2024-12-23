 
from itertools import product
from sklearn.model_selection import KFold
from constants_config import TARGET_VARIABLES

import numpy as np


def trainingInit(model,target_variable_name=None):
    
        """
        model: a baseModel Object (can be PLSR, XGBoost and RandomForest)
        target_variable_name: the name of the target_variable (if not using multioutput.)
        """

        if model.is_multi_output and target_variable_name is not None:
            rmse_results = {target_var: [] for target_var in TARGET_VARIABLES}
        else:
            rmse_results = {target_variable_name: []}
        
        return rmse_results


def createHyperParametersCombinations(param_grid):
     
    # This function creates hyper parameter combination - easier to loop on.
    values = param_grid.values()
    param_combos = {}
    combinations = list(product(*values))

    for i,comb in enumerate(combinations):
        param_combos[f'combination_{i}'] = comb
    
    return param_combos



def tuningLoop(model,target_variable_name,params,PLSR_Tuning):
     rmse_results = trainingInit(model, target_variable_name)

     if model.is_multi_output:
            n_value, sc_value, st_value = TARGET_VARIABLES

     if PLSR_Tuning:
        for n_components in params:
            model.set_params(n_components)
            rmse = model.validate()

            if model.is_multi_output:
                 n_value_rmse,sc_value_rmse, st_value_rmse = rmse
                 avg_rmse = np.mean(rmse)
                 # Mainly For plotting
                 rmse_results[n_value] += [n_value_rmse]
                 rmse_results[sc_value] += [sc_value_rmse]
                 rmse_results[st_value] += [st_value_rmse]
                 # For choosing the best hyper parameters
                 rmse_results['Avg_RMSE'] = (model.get_params(),avg_rmse)
        else:
            rmse_results[target_variable_name] = (model.get_params(),rmse)

     else:
        
        for hyperparams in params.values():
             # hyperparams is a dictionary
             # combination is the number of the combination
             model.set_params(*hyperparams)
             rmse = model.validate()

             if model.is_multi_output:
                n_value_rmse,sc_value_rmse, st_value_rmse = rmse
                avg_rmse = np.mean(rmse)
                # Mainly For plotting
                rmse_results[n_value] += [n_value_rmse]
                rmse_results[sc_value] += [sc_value_rmse]
                rmse_results[st_value] += [st_value_rmse]
                # For choosing the best hyper parameters
                rmse_results['Avg_RMSE'] = (model.get_params(),avg_rmse)

             else:
                 rmse_results[target_variable_name] = (model.get_params(),rmse)
     return rmse_results
             
                
# Use this function for tuning your model's hyper parameters.         
def hyperParameterTuning(model,target_variable_name=None, PLSR_Tuning = False):
    """
    model: a baseModel Object (can be PLSR, XGBoost and RandomForest)
    target_variable_name: the name of the target_variable (if not using multioutput.)
    """

    if not PLSR_Tuning:
          # For random forest and XGBoost always create combinations
          params = createHyperParametersCombinations(model.param_grid)
    else:
          # For PLSR dont create - only one hyperparameter
          params = list(model.param_grid.values())
    
    rmse_results  = tuningLoop(model,target_variable_name,params,PLSR_Tuning)

    return rmse_results




# Use this function after you've finished optimizing your model!

def CV10(model,target_variable_name=None, n_splits=10):
        
        """
        model: a baseModel Object (can be PLSR, XGBoost and RandomForest) - With the best Hyper Parameters.
        target_variable_name: the name of the target_variable (if not using multioutput.)
        """

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        rmse_results = trainingInit(model, target_variable_name)

        if model.is_multi_output:
            n_value, sc_value, st_value = TARGET_VARIABLES

        for train_index, val_index in kf.split(model.dataset.X_train):

            X_train_fold, X_val_fold = model.dataset.X_train[train_index], model.dataset.X_train[val_index]
            y_train_fold, y_val_fold = model.dataset.Y_train[train_index], model.dataset.Y_train[val_index]

            rmses = model.CrossValidate(X_train_fold,y_train_fold, X_val_fold, y_val_fold)

            if model.is_multi_output:
                n_rmse,sc_rmse,st_rmse = rmses
                rmse_results[n_value] += [n_rmse]
                rmse_results[sc_value] += [sc_rmse]
                rmse_results[st_value] += [st_rmse]
            else:
                dependent_feature_rmse = rmses 
                rmse_results[target_variable_name] += [dependent_feature_rmse]
        
        return rmse_results 



     


     



    
