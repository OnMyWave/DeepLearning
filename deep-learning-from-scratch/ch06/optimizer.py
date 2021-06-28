import numpy as np

# 1. SGD (Stochastic Gradient Descent)

class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr
    
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 2. Momentum : 공이 미끄러지는 듯한 이미지

class Momentum: 
    def __init__(self,lr = 0.01,momentum = 0.9) :
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key] 

# 3. AdaGrad : 학습률을 큰 쪽에서 작아지는 쪽으로 조절

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# 4. Adam : Momentum + AdaGrad

