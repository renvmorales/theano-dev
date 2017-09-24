# This API implements the following tasks:
#  - Multi-layer 'relu' ANN models for multiclass problems
#  - Standard GD optimization 
# 
# Implementation using Theano library and Numpy.
import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt 
import time
from  ann_functions import *
from sklearn.utils import shuffle




class ANN_relu(object):
	def __init__(self, M):
		# this assures that all hidden unities are stored in a list
		if isinstance(M, int):
			self.M = [M]  # in case there is a single hidden layer...
		else:
			self.M = M



	def fit(self, X, Y, alpha=1e-3, reg=1e-4, epochs=5000, show_fig=False,
		Nbatch=10):
		N, D = X.shape
		K = len(np.unique(Y))
		batch_sz = int(N/Nbatch)  # batch size

		self.N = N  # this variable will be used for normalization
		self.D = D  # store the dimension of the training dataset
		self.K = K  # output dimension
		self.batch_sz = batch_sz

		# stores all hyperparameter values
		self.hyperparameters = {'alpha':alpha, 'reg':reg, 'epochs':epochs,
		'Nbatch': Nbatch}


		# creates an indicator matrix for the target
		Trgt = np.zeros((N, K))
		Trgt[np.arange(N), Y.astype(np.int32)] = 1


		# creates a list with the number of hidden unities (+ input/output)
		hdn_unties = [D] + self.M + [K]
		W_init = []
		b_init = []
		# initializes all weights randomly
		for k in range(1,len(self.M)+2):
			W, b = init_weights(hdn_unties[k-1], hdn_unties[k])
			W_init.append(W.astype(theano.config.floatX))
			b_init.append(b.astype(theano.config.floatX))



		# placeholders for theano variables
		thX = T.matrix()
		thT = T.matrix()
		W = []  # a list for theano shared variables
		b = []
		for i in range(len(W_init)):
			W.append(theano.shared(W_init[i]).astype(theano.config.floatX))
			b.append(theano.shared(b_init[i]).astype(theano.config.floatX))



		# symbolic feed-forward expression 
		thZ = [thX]
		for i in range(0,len(self.M)):
			thZ.append(T.nnet.relu(thZ[i].dot(W[i])+b[i]))
		thZ.append(T.nnet.softmax(thZ[len(self.M)].dot(W[len(self.M)])+b[len(self.M)]))
		thY = thZ[-1]



		# symbolic expression for the cost function
		J = T.mean(T.nnet.categorical_crossentropy(thY, thT))
		# J = -1/batch_sz*T.sum((T.log(thY[T.arange(batch_sz), 
		# 	T.argmax(thT, axis=1)])))


		# symbolic expression for computing a prediction
		prediction = T.argmax(thY, axis=1)


		# symbolic expression for the update equations (back-prop)
		updt_W = []
		updt_b = []
		for i in range(len(W)):
			updt_W.append(W[i] - alpha*(T.grad(J, W[i]) + reg/(2*batch_sz)*W[i]))
			updt_b.append(b[i] - alpha*(T.grad(J, b[i]) + reg/(2*batch_sz)*b[i]))


		# generate a list of tuples with weights and the update expression
		param = []
		for i in range(len(W)):
			param.append(W[i])
			param.append(b[i])

		upd_param = [] 
		for i in range(len(W)):
			upd_param.append(updt_W[i])
			upd_param.append(updt_b[i])

		ls_updt = [(param[i], upd_param[i]) for i in range(len(param))]



		# define a training function for the model using updating rules
		# (produces 2 outputs - cost and prediction)
		train = theano.function(
			inputs=[thX, thT],
			outputs=[J, prediction],
			updates=ls_updt
		)
		


		# this function only returns a prediction for the model 
		# (should be used as part of another method)
		self.get_prediction = theano.function(
			inputs=[thX],
			outputs=prediction
		)


		
		cost=[]
		start = time.time()	# <-- starts measuring the optimization time from this point on...			

		# optimization loop
		for i in range(int(epochs/Nbatch)):
			Xbuf, Trgtbuf = shuffle(X, Trgt)
			for j in range(int(Nbatch)):
				X_b = Xbuf[(j*batch_sz):(j*batch_sz+batch_sz),:] # input batch sample
				Trgt_b = Trgtbuf[(j*batch_sz):(j*batch_sz+batch_sz),:] # output batch sample
				cost_val, prediction_val = train(X_b, Trgt_b)
				accur = np.mean(self.predict(X)==Y)
				cost.append(cost_val)
				if (i*Nbatch+j+1) % 100 == 0:
					print("Epoch: %d  Cost: %.4f  Accuracy: %.4f" % (i*Nbatch+j+1, cost_val, accur))


		end = time.time()
		self.elapsed_t = (end-start)/60 # total elapsed time
		self.cost = cost # stores all cost values 

		print('\nOptimization complete')
		print('\nElapsed time: {:.3f} min'.format(self.elapsed_t))


		# customized plot with the resulting cost values
		if show_fig: 
			plt.plot(cost, label='Cost function J')
			plt.title('Evolution of the Cost through a batch GD optimization in Theano    Total runtime: {:.3f} min'.format(self.elapsed_t)+'    Final Accuracy: {:.3f}'.format(np.mean(Y==self.predict(X))))
			plt.xlabel('Epochs')
			plt.ylabel('Cost')
			plt.legend()
			plt.show()




	def predict(self, X):
		return self.get_prediction(X)







def main():
# number of samples for each class
	N_class = 1000 

# generate random 2-D points 
	X1 = np.random.randn(N_class,2)+np.array([2,2])
	X2 = np.random.randn(N_class,2)+np.array([-2,-2])
	X3 = np.random.randn(N_class,2)+np.array([-2,2])
	X4 = np.random.randn(N_class,2)+np.array([2,-2])
	X = np.vstack([X1,X2,X3,X4])

# labels associated to the input
	Y = np.array([0]*N_class+[1]*N_class+[2]*N_class+[3]*N_class)
	# Y = np.reshape(Y, (len(Y),1))


# general data information for the training process
	print('Total input samples:',X.shape[0])
	print('Data dimension:',X.shape[1])
	print('Number of output classes:',len(np.unique(Y)))
	print('\n')


# scatter plot of original labeled data
	plt.scatter(X[:,0],X[:,1],c=Y,s=50,alpha=0.5)
	plt.show()



# create an ANN model with the specified 4 hidden layers
	model = ANN_relu([10,10,10,10])


# fit the model with the hyperparameters set	
	model.fit(X, Y, alpha=1e-2, epochs=5000, reg=0.01, Nbatch=20, 
		show_fig=True)



# compute the model accuracy	
	Ypred = model.predict(X)
	print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))







if __name__ == '__main__':
	main()