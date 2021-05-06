from summary import forward, init_network
from dataset.mnist import load_mnist
import numpy as np

def get_data():
    (x_train, t_train), (x_test,t_test) = load_mnist(normalize=True,flatten=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network 

x,t = get_data()
network = init_network()

accuracy_cnt = 0 
for i in range(len(x)):
    y = forward(network,i)
    p = np.argmax(y)
    if p == t[i] :
        accuracy_cnt += 1

print("accuracy is " + str(accuracy_cnt / len(x)))

