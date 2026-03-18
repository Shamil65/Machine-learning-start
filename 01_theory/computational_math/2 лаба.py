import numpy as np

A = np.array([
    [1.4, 5.6, 1.9],
    [0.7, -0.5, 2.4],
    [3.6, -3.1, 2.8]
])
b = np.array([10.8, 3.6, 6.1])

x = np.zeros(3)
eps = 1e-6
max_iter = 50

for k in range(max_iter):
    x_old = x.copy()
    x[0] = (b[0] - A[0,1]*x[1] - A[0,2]*x[2]) / A[0,0]
    x[1] = (b[1] - A[1,0]*x[0] - A[1,2]*x[2]) / A[1,1]
    x[2] = (b[2] - A[2,0]*x[0] - A[2,1]*x[1]) / A[2,2]
    
    if np.linalg.norm(x - x_old, np.inf) < eps:
        break

print("x1 =", x[0])
print("x2 =", x[1])
print("x3 =", x[2])