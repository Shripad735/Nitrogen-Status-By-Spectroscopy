import os
import pandas as pd
import sys
sys.path.append('../')
from constants_config import DATA_FOLDER_PATH, TARGET_VARIABLES, NON_FEATURE_COLUMNS


class Dataset:
    def __init__(self, train_file_name, validation_file_name, test_file_name):
        self.ID_train, self.X_train, self.Y_train = self.splitDependentAndIndependent(train_file_name)
        self.ID_val, self.X_val, self.Y_val = self.splitDependentAndIndependent(validation_file_name)
        self.ID_test, self.X_test, self.Y_test = self.splitDependentAndIndependent(test_file_name)

    def splitDependentAndIndependent(self, data_file_name):
        data = pd.read_parquet(os.path.join(DATA_FOLDER_PATH, data_file_name))
        id = data['ID']
        X = data.drop(columns=NON_FEATURE_COLUMNS)
        Y = data[TARGET_VARIABLES]
        return id, X, Y
