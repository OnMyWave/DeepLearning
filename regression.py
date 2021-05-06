import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline
#data points in column vector [input, output]
x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8, 3.0, 4.0, 4.3, 4.4, 4.9]).reshape(
    -1, 1 
)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8, 2.7, 3.0, 3.5, 3.7, 3.9]).reshape(
    -1, 1
)

plt.figure(figsize=(10, 8))
plt.plot(x, y, "ko")
plt.title("Data", fontsize=15)
plt.xlabel("X", fontsize=15)
plt.ylabel("Y", fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()

m = y.shape[0]
# A = np.hstack([x, np.ones([m, 1])])
A = np.hstack([x ** 0, x])
A = np.asmatrix(A)

theta = (A.T * A).I * A.T * y

print("theta:\n", theta)
# to plot
plt.figure(figsize=(10, 8))
plt.title("Regression", fontsize=15)
plt.xlabel("X", fontsize=15)
plt.ylabel("Y", fontsize=15)
plt.plot(x, y, "ko", label="data")

# to plot a straight line (fitted line)
xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp = theta[0, 0] + theta[1, 0] * xp

plt.plot(xp, yp, "r", linewidth=2, label="regression")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()