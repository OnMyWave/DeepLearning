# Chatper2. 퍼셉트론

## 논리 회로
# 가중치와 편향을 더한 AND 게이트
def AND(x1,x2):
    w1, w2, bias = 0.3, 0.5, -0.7
    y = w1*x1 + w2*x2 + bias
    if y >= 0 :
        return 1
    else: 
        return 0

print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))
print()

# 가중치와 편향을 더한 OR 게이트
def OR(x1,x2):
    w1, w2 , bias = 0.3, 0.5, -0.2
    y = w1*x1 + w2*x2 + bias
    if y >=0 :
        return 1
    else:
        return 0

print(OR(0,0))
print(OR(1,0))
print(OR(0,1))
print(OR(1,1))
print()


def NAND(x1,x2):
    w1, w2, bias = -0.3, -0.5, 0.7
    y = w1*x1 + w2*x2 + bias
    if y >=0 :
        return 1
    else:
        return 0

print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))
print()

def XOR(x1,x2):
    x3 = NAND(x1,x2)
    x4 = OR(x1,x2)
    return AND(x3,x4)

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
print()


### numpy 이용
print("It's Numpy Time")
print()

import numpy as np

def numpy_AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.3,0.5])
    bias = - 0.7
    theta = np.sum(x*w) + bias
    if theta > 0 : 
        return 1
    else:
        return 0

def numpy_OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.3,0.5])
    bias = -0.2
    theta = np.sum(x*w) + bias
    if theta >= 0 :
        return 1
    else: 
        return 0

def numpy_NAND(x1,x2):
    x = np.array([x1,x2])
    w = -np.array([0.3,0.5])
    bias = 0.7
    theta = np.sum(w*x) + bias
    if theta >= 0:
        return 1
    else:
        return 0

print(numpy_AND(0,0))
print(numpy_AND(1,0))
print(numpy_AND(0,1))
print(numpy_AND(1,1))
print()


print(numpy_OR(0,0))
print(numpy_OR(1,0))
print(numpy_OR(0,1))
print(numpy_OR(1,1))
print()


print(numpy_NAND(0,0))
print(numpy_NAND(1,0))
print(numpy_NAND(0,1))
print(numpy_NAND(1,1))
print()

