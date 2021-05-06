import numpy as np

def softmax1(a): # 이러한 방식은 큰 수가 들어갔을 때 overflow가 발생 !
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)

def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    return exp_a / np.sum(exp_a)

a = np.array([0.3,0.4,4.0])
print(softmax2(a))
    
    