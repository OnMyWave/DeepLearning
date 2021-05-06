# Chapter4. 
import numpy as np

# 평균 제곱 오차 : mean-squared-error
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

print(mean_squared_error(np.array(y),np.array(t)))

# Cross-Entropy-error

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

print(cross_entropy_error(np.array(y),np.array(t)))

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train ), (x_test, t_test) = load_mnist(normalize=True,flatten=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

def batch_cross_entropy(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))/n
