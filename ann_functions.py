# A set of useful functions for training and forward propagation  
# of artificial neural nets
import numpy as np 





# initializes with random weights
def init_weights(R, S):
	W = np.random.randn(R,S)/np.sqrt(R+S) # Xavier initialization
	b = np.random.rand(S)
	# b = np.zeros(S)
	return W.astype(np.float32), b.astype(np.float32)



# sigmoid function
def sigmoid(x):
	return 1 / (1+np.exp(-x))


# linear rectifier function
def relu(x):
	return x*(x>0)


# softmax function for multiclass problems
def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)


# defines the cross-entropy cost function for binary classification
def cross_entropy_bin(T, pY):
	N = len(T)  # normalizing factor
	return -1/N*(T*np.log(pY) + (1-T)*np.log(1-pY)).sum()


# defines the cross-entropy cost function for multiclass
def cross_entropy_multi(T, pY):
	N = len(T)
	return -1/N*(T*np.log(pY)).sum()


# same as 'cross_entropy_multi' but cost computation is faster when 
# dealing with large dimensional datasets
def cross_entropy_multi2(T, pY):	
	N = len(T)
	return -1/N*np.log(pY[np.arange(N), np.argmax(T, axis=1)]).sum()


# defines the residual squares sum cost function for regression 
def resid_squares(TRGT, Y):
	return 0.5*sum((TRGT-Y)**2)





