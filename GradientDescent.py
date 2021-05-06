import numpy as np
H=np.array([[2,0],[0,2]])
g=-np.array([[6],[6]])
x=np.zeros((2,1))
alpha=0.2
for i in range(50):
    df= H*x + g
    x= x - alpha*df

print(x)