from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from FoldCreator import FoldCreator

import numpy as np

class RF():

	def __init__(self, n_estimators, max_depth, min_samples_leaf, min_samples_split):

		self.params = {
		'n_estimators':n_estimators,
		'max_depth': max_depth,
		'min_samples_leaf':min_samples_leaf,
		'min_samples_split':min_samples_split
		}

		self.initModel()

	def train(self,x_train,y_train):
		self.model.fit(x_train,y_train)

	def predict(self,x_test):
		self.preds = self.model.predict(x_test)
		return self.preds

	def computeMetrics(self,y_test):
		f1 = f1_score(y_test,self.preds)
		oob_error = 1 - self.model.oob_score_
		return f1,oob_error

	def initModel(self):
		self.model = RandomForestClassifier(**self.params,oob_score=True)



class RandomForestCV():

	def __init__(self,model,folds,x,y,):

		self.x = x
		self.y = y
		self.n_folds = folds
		self.model = model

		self.fold_creator = FoldCreator(self.n_folds,self.x,self.y)


	def runCV(self):
		f1s = []
		oobs = []
		fold_indicies = self.fold_creator.GetFoldIndicies()

		for fold in range(self.n_folds):
			
			train,test = 0,1
			train_indicies,test_indicies = fold_indicies[fold][train],fold_indicies[fold][test]

			x_train,y_train= self.x[train_indicies],self.y[train_indicies]
			x_test,y_test = self.x[test_indicies], self.y[test_indicies]

			f1,oob = self.computeRFPerformance(x_train,y_train, x_test, y_test)

			f1s.append(f1)
			oobs.append(oob)

		# Gives f1 
		self.avgRFF1 = np.mean(f1s)
		# Gives generalization error..
		self.avgRFoobErr = np.mean(oobs)
		return self.avgRFF1, self.avgRFoobErr


	def computeRFPerformance(self,x_train,y_train,x_test,y_test):
		self.model.train(x_train,y_train)
		self.model.predict(x_test)
		f1,oob_error = self.model.computeMetrics(y_test)
		return f1,oob_error
