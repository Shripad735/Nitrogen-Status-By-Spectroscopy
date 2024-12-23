from Dataset import Dataset

from sklearn.metrics import root_mean_squared_error

# Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor


from constants_config import TARGET_VARIABLES

train_file_name = 'train_data.parquet'
validation_file_name = 'validation_data.parquet'
test_file_name = 'test_data.parquet'


dataset = Dataset(train_file_name,validation_file_name,test_file_name)


class BaseModel:

    def __init__(self,dataset,model,param_grid, is_multi_output=False):

        self.dataset = dataset
        self.model = model
        self.is_multi_output = is_multi_output
        self.param_grid = param_grid
        self.best_params = None

        self.random_state = 42

    
    def computeRMSE(self, Y,Y_hat):
        """
        Just a method to compute RMSE
        """
        
        if self.is_multi_output:
            n_value_rmse,sc_value_rmse,st_value_rmse = root_mean_squared_error(y_true=Y,y_pred=Y_hat, multioutput='raw_values')
            return n_value_rmse, sc_value_rmse, st_value_rmse
        
        return root_mean_squared_error(y_true=Y,y_pred=Y_hat)


    def CrossValidate(self,x_train,y_train,x_val,y_val):
        self.model.fit(x_train,y_train)
        y_hat = self.model.predict(x_val)

        if self.is_multi_output:
            n_value_rmse,sc_value_rmse,st_value_rmse = self.computeRMSE(y_val, y_hat)
            return n_value_rmse, sc_value_rmse, st_value_rmse
        
        return  self.computeRMSE(y_val, y_hat)
    
            
    def validate(self):
        """
        Can be used for Hyperparameter Tuning.
        """
        self.model.fit(self.dataset.X_train,self.dataset.Y_train)
        y_hat = self.model.predict(self.dataset.X_val)
        
        return  self.computeRMSE(self.dataset.Y_val, y_hat)
    
    def evaluate(self):
        """
        Use this for testing only! (After training the model using CV10)
        """
        y_hat = self.model.predict(self.dataset.X_test)

        return self.computeRMSE(self.dataset.Y_test,y_hat)



# Make sklearn models inherit base model class
class XGBoostModel(BaseModel):
    def __init__(self, dataset,param_grid=None, is_multi_output=False):

        if is_multi_output:
            model = MultiOutputRegressor(XGBRegressor())
        else:
            model = XGBRegressor()

        super().__init__(dataset,model,param_grid, is_multi_output)

class RandomForestModel(BaseModel):
    def __init__(self, dataset,param_grid=None, is_multi_output=False):

        if is_multi_output:
            model = MultiOutputRegressor(RandomForestRegressor())
        else:
            model = RandomForestRegressor()

        super().__init__(dataset,model,param_grid, is_multi_output)

class PLSRModel(BaseModel):
    def __init__(self, dataset,param_grid=None, is_multi_output=False):

        if is_multi_output:
            model = MultiOutputRegressor(PLSRegression())
        else:
            model = PLSRegression()
        super().__init__(dataset, model, param_grid, is_multi_output)









        

            



