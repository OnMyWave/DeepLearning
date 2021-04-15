# Chapter3. 신경망

import numpy as np
import matplotlib.pyplot as plt

# 3.2.2 계단 함수 구현하기

def step_function(x):
    if x>=0:
        return 1
    else:
        return 0

def step_function2(x):
    y = x > 0 
    return y.astype(np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_function2(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# 3.2.4 시그모이드 함수 구현하기

def sigmoid(x):
    return 1/(1+np.exp(-x))

y2 = sigmoid(x)
plt.plot(x,y2)
plt.ylim(-0.1,1.1)
plt.show()

#3.2.7 ReLU 함수 구현하기

def ReLU(x):
    return np.maximum(0,x)  # 실수가 아닌 Numpy 객체들도 받기 위함
    #maximum 함수는 객체 중에 큰 거 반환

print(ReLU(x))

def indentity_function(x):
    return x

# Softmax 함수 구현하기
def softmax(x):
    max_num = np.max(x)
    exp_x = np.exp(x-max_num)
    return exp_x/np.sum(np.exp(x - max_num))
    # np.max는 최대값 반환

# 3.4 3층 신경망 구현하기

X = np.array([1.0,0.5])
W = np.array([[0.3,0.5],[0.25,0.3],[0.1,0.5],[0.4,0.1],[0.2,0.5]])
dot_WX = np.dot(W,X)
z = sigmoid(dot_WX)

print(W)
print(dot_WX)
print(z)
print()
print()
print()

# 구현 정리

def init_network():
    network = {}
    network['w1'] = np.array([[0.3,0.5],[0.25,0.3],[0.1,0.5],[0.4,0.1],[0.2,0.5]])
    network['w2'] = np.array([[0.4,0.1,0.2,0.4,0.5],[0.3,0.7,0.4,0.9,0.1]])
    network['b1'] = np.array([0.1,0.1,0.1,0.2,0.3])
    network['b2'] = np.array([0.2,0.1])
    
    return network

print(init_network())
print()


def forward(network,x):
    y1 = np.dot(network['w1'],x) + network['b1']
    z1 = sigmoid(y1)
    y2 = np.dot(network['w2'],z1) + network['b2']
    z2 = sigmoid(y2)
    z = indentity_function(z2)
    return z

network = init_network()

X = np.array([0.3,0.5])
print(forward(network,X))
print(softmax(X))
