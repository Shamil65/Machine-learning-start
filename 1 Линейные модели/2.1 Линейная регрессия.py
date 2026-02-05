import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.01, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return "MyLineReg class: n_iter={}, learning_rate={}".format(self.n_iter, self.learning_rate)
    
    def fit(self, X, y, verbose=False):
        X.insert(0, "bias", 1)
        print(X)

        print(X.shape)
        m, n = X.shape
        print(n, m)

        self.weights = np.ones(n)
        print(self.weights)

        X_mat = X.values
        print(X_mat)
        y_vec = y.values
        print(y_vec)
        y_pred = X_mat @ self.weights
        errors = y_pred - y_vec
        errors_2 = errors**2
        MSE = np.mean(errors**2)
        print(y_pred)
        print(errors_2)
        print(errors)
        print(MSE)

        for i in range(1, self.n_iter + 1):
            pass
            # y_pred = X_mat @ self.weights
            # errors = y_pred - y_vec
            # print()       


X = pd.DataFrame({"x1": [1, 2, 3],
                  "x2": [4, 5, 6],
                  "x3": [7, 8, 9]})
y = pd.Series([1, 2, 3])

verbose = 10
MyLineReg1 = MyLineReg()
print(MyLineReg1)
MyLineReg1.fit(X, y, verbose)