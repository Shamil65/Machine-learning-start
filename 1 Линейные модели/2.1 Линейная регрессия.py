import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.01, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return "MyLineReg class: n_iter={}, learning_rate={}".format(self.n_iter, self.learning_rate)
    
    def demo_fit(self, X, y, verbose=False):

        print("X до ввода нового столбца: \n", X, "\n")
        X.insert(0, "bias", 1)
        m, n = X.shape # m - кол-во строк; n - кол-во столбцов

        print("self.weights до заполнения: \n", self.weights, "\n")
        self.weights = np.ones(n) # засовывваем в эту переменную np массив со значениями 1 в кол-ве равном кол-ву переменных в одном векторе в X
        
        X_mat = X.values # Превращаем из DataFrame в np array
        y_vec = y.values
        y_pred = X_mat @ self.weights

        errors = y_pred - y_vec
        MSE = np.mean(errors**2)

        # gradient = np.sum(2*(y_pred - y_vec) @ X_mat)
        # gradient = 2*(y_pred - y_vec) @ X_mat
        # print(gradient)

        print("X после ввода нового столбца: \n", X, "\n")
        print("self.weights после заполнения: \n", self.weights, "\n")
        print("X_mat: \n", X_mat, "\n")
        print("y_vec: \n", y_vec, "\n")
        print("y_pred \n", y_pred, "\n")
        print("errors \n", errors, "\n")
        print("MSE \n", MSE, "\n")




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
MyLineReg1.demo_fit(X, y, verbose)