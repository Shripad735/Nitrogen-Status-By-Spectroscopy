from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class MultiOutputPLSRegression:

    def __init__(self, n_splits=10, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.multiPLSR = None

    def evaluatePerformance(self, X_train, Y_train, X_val, Y_val):

        """
        Evaluates the performance of the PLSR model on the validation set and returns the RMSE 

        pls - PLSR model (sklearn)

        X_train - numpy array, training data
        Y_train - numpy array, training dependent variable

        X_val - numpy array, validation data
        Y_val - numpy array, validation dependent variable

        """
        if self.multiPLSR is None:
            print(f"Define the model before evaluating it's performance")
            return
        
        self.multiPLSR.fit(X_train,Y_train)
        Y_pred = self.multiPLSR.predict(X_val)

        # Compute RMSE
        mse = mean_squared_error(Y_val, Y_pred,multioutput='uniform_average')
        rmse = np.sqrt(mse)
        return rmse
    

    def findOptimalNumberOfComponents(self, X_train, Y_train,X_val,Y_val, maximum_components):
        """
        Finds the optimal number of components to use in the PLSR model


        X_train - numpy array, training data 
        Y_train - numpy array, training dependent variable

        X_val - numpy array, validation data
        Y_val - numpy array, validation dependent variable

        maximum_components - int, number of components to use in the PLSR model

        """

        rmse_list = []
        n_components_list = list(range(1,maximum_components + 1))
        self.optimization_metric_results = {}

        for n_components in tqdm( n_components_list,
                                 desc='Optimizing Number of Components', 
                                 total=len(n_components_list)):
            # Init PLSR
            pls = PLSRegression(n_components=n_components)
            self.multiPLSR = MultiOutputRegressor(pls)

            # Evaluate performance on the validation set
            rmse = self.evaluatePerformance(X_train, Y_train, X_val, Y_val)

            # Store the results
            self.optimization_metric_results[n_components] = rmse 

            rmse_list.append(rmse)
        
        # Will be switch anyways...
        self.multiPLSR = None
            
        return rmse_list
    
    def plotMetricResults(self):
        """
        Plots the results of the optimization metric 
        """

        # Extract the number of components and RMSE values
        n_components = list(self.optimization_metric_results.keys())
        rmse_values = [self.optimization_metric_results[key] for key in n_components]

        # Plot the RMSE values
        plt.plot(n_components, rmse_values, label='RMSE')

        # Mark the minimum RMSE value with a red dot
        min_rmse = min(rmse_values)
        min_rmse_key = n_components[rmse_values.index(min_rmse)]
        plt.plot(min_rmse_key, min_rmse, 'ro', label=f'Min RMSE at {min_rmse_key} components')

        # Formatting plot
        plt.xlabel('Number of Components')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE')
        plt.legend()
        plt.title('Optimization Metric: RMSE')
        plt.tight_layout()
        plt.show()

    
   
    def crossValidation(self, X_train, Y_train, n_components):
        """
        Performs CV-10 on the training data.

        For training only!!! So use this after you have selected the optimal number of components

        X_train - numpy array, training data 
        Y_train - numpy array, training dependent variable

        n_components - int, number of components to use in the PLSR model (optimal)
        n_splits - int, number of splits to use in the KFold cross validation

        """

        # Initialize the PLSR model with the optimal number of components
        pls = PLSRegression(n_components=n_components)
        self.multiPLSR = MultiOutputRegressor(pls)

        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)

        rmses = []
        
        for train_index, val_index in tqdm(kf.split(X_train), desc='Cross Validation', total=self.n_splits):

            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

            rmse_fold = self.evaluatePerformance(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold)

            rmses.append(rmse_fold)

        return rmses 




