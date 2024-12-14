import numpy as np 

class FoldCreator():

	def __init__(self,n_folds,x,y):

		self.x = x
		self.y = y
		self.n_folds = n_folds

	def GetFoldIndicies(self):
		datasize = self.x.shape[0]
		foldsize = self.n_folds
		indicies = np.arange(datasize)


		np.random.shuffle(indicies)
		
		folds_indicies = []

		for i in range(self.n_folds):
			start = i*foldsize
			end = (i+1)*foldsize

			test_indicies = indicies[start:end]
			train_indicies = np.concatenate([indicies[:start], indicies[end:]])

			folds_indicies.append((train_indicies,test_indicies))

		return folds_indicies




