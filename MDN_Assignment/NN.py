import numpy as np
import torch 
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from FoldCreator import FoldCreator


# Neural Network Class
class NN(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):

		super().__init__()
		self.layer1 = nn.Linear(input_size,hidden_size)
		self.layer2 = nn.Linear(hidden_size,output_size)
		self.relu = nn.ReLU()


	def forward(self,x):
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		return x 


def createDataLoader(x,y,batch_size,shuffle=False):
	temp_dataset = TensorDataset(x,y)
	data_loader = DataLoader(temp_dataset, batch_size, shuffle=shuffle)

	return data_loader

# Training & Validation Phase
def trainNN(model,x_train, y_train,
			optimizer,
			criterion,
			train_batch_size,
			val_batch_size = None,
			x_val=None,y_val=None,validate = False,epochs=5):
	
	
	train_loader = createDataLoader(x_train,y_train,train_batch_size, shuffle=True)

	if validate:
		val_loader = createDataLoader(x_val,y_val,val_batch_size)
		val_losses = []

	model.train()
	total_samples = 0 
	train_losses = []
	
	for _ in range(epochs):
		total_samples = 0 
		running_loss = 0
		# Training Phase
		for _ , (x_train,y_train) in enumerate(train_loader):

			outputs = model(x_train)
			optimizer.zero_grad()
			loss = criterion(outputs,y_train)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()*x_train.size(0)
			total_samples += x_train.size(0)
	
		train_losses.append(running_loss/total_samples)

		# Validating phase
		if validate:

			val_total_samples = 0
			val_running_loss = 0
			model.eval()

			with torch.no_grad():
				for _, (x_val,y_val) in enumerate(val_loader):

					val_outputs = model(x_val)
					val_loss = criterion(val_outputs, y_val)
					val_running_loss += val_loss.item()*x_val.size(0)
					val_total_samples += x_val.size(0)
				
				val_losses.append(val_running_loss/val_total_samples)
			model.train()

		if validate:
			return np.mean(train_losses), np.mean(val_losses)
		
		return np.mean(train_losses)
					

# Testing Phase
def testNN(model,x_test,y_test,batch_size,criterion):

	model.eval()
	correct = 0
	test_loss = 0
	total = 0

	test_loader = createDataLoader(x_test,y_test,batch_size,shuffle=False)
	
	f1s = []
	with torch.no_grad():
		for _ , (x_test, y_test) in enumerate(test_loader):
			outputs = model(x_test)
			predicted = (outputs > 0.5).float()
			test_loss  += criterion(outputs,y_test).item()*y_test.size(0)
			total += x_test.size(0)
		
			f1 = f1_score(y_test,predicted)
			f1s.append(f1)
	test_loss = test_loss/total

	return np.mean(f1s) ,test_loss


# NeuralNetworkCV
class NeuralNetworkCV():

	def __init__(self,x,y,n_folds,model):

		self.x = x
		self.y = y
		self.n_folds = n_folds
		self.fold_creator = FoldCreator(self.n_folds,self.x,self.y)
		self.model = model


	def runCV(self,optimizer,
			criterion,):

		f1s = []
		loss = []

		fold_indicies = self.fold_creator.GetFoldIndicies()

		for fold in range(self.n_folds):

			train_indicies,test_indicies = fold_indicies[fold][0],fold_indicies[fold][1]

			x_train,y_train= self.x[train_indicies],self.y[train_indicies]
			x_test,y_test = self.x[test_indicies], self.y[test_indicies]

			_ = trainNN(self.model,x_train,y_train, 
						optimizer,criterion, 900)

			f1,test_loss = testNN(self.model,x_test,y_test, 100,criterion)
			f1s.append(f1)
			loss.append(test_loss)

		self.nnAvgf1 = np.mean(f1s)
		self.nnAvgLoss = np.mean(loss)

		return self.nnAvgf1, self.nnAvgLoss
		









