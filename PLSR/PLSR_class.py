import sys 
sys.path.append('../baseline_for_training/')

from baseModels import BaseModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor

class PLSRModel(BaseModel):

    def __init__(self, dataset,param_grid=None, is_multi_output=False):
        if is_multi_output:
            model = MultiOutputRegressor(PLSRegression())
        else:
            model = PLSRegression()
        super().__init__(dataset, model, param_grid, is_multi_output)
        
   