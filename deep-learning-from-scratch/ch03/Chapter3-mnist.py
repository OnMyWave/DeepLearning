import sys
sys.path.append("/Users/onmywave/Desktop/Github/DeepLearning/deep-learning-from-scratch/")
from dataset.mnist import load_mnist
from Chapter3 import sigmoid, softmax
import numpy as np
import pickle


#
def get_data():
    (x_train, t_train) ,(x_test,t_test) = load_mnist(flatten= True,
    normalize=True,one_hot_label=True)
    return x_test,t_test

def init_network():
    with open("/Users/onmywave/Desktop/Github/DeepLearning/deep-learning-from-scratch/deep-learning-from-scratch-master/ch03/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

x,t = get_data()

network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == np.argmax(t[i]):
        accuracy_cnt += 1

print("ACCURACY : " + str(float(accuracy_cnt/len(x))))