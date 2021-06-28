from http.client import BadStatusLine
import sys
from typing import NoReturn
sys.path.append("/Users/onmywave/Desktop/Github/DeepLearning/deep-learning-from-scratch/")
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet
import numpy as np

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Hyperparameters
iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.05

# repeat / epoch
iter_per_epoch = max(train_size/batch_size,1)
network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch,t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate*grad[key]
    
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Train acc, Test acc : " + str(train_acc)+' , '+str(test_acc))
