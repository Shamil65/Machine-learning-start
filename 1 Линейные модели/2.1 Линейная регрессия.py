import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.001, weights=None):
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

        gradient = (2 / m) * (X_mat.T @ (y_pred - y_vec))
        

        print("X после ввода нового столбца: \n", X, "\n")
        print("self.weights после заполнения: \n", self.weights, "\n")
        print("X_mat: \n", X_mat, "\n")
        print("X_mat.T: \n", X_mat.T, "\n")
        print("y_vec: \n", y_vec, "\n")
        print("m, n: \n", m, n, "\n")
        print("y_pred \n", y_pred, "\n")
        print("errors \n", errors, "\n")
        print("MSE \n", MSE, "\n")
        print("gradient \n", gradient, "\n")




    def fit(self, X, y, verbose=False):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        m, n = X.shape # m - кол-во строк; n - кол-во столбцов

        self.weights = np.ones(n) # засовывваем в эту переменную np массив со значениями 1 в кол-ве равном кол-ву переменных в одном векторе в X
        
        X_mat = X.values # Превращаем из DataFrame в np array
        y_vec = y.values
        y_pred = X_mat @ self.weights

        errors = y_pred - y_vec
        MSE = np.mean(errors**2)

        if verbose:
            print(f"start | loss: {MSE}")

        for i in range(1, self.n_iter + 1):
            y_pred = X_mat @ self.weights
            errors = y_pred - y_vec           # 🔹 ОБЯЗАТЕЛЬНО обновляем
            gradient = (2 / m) * (X_mat.T @ errors)
            self.weights = self.weights - self.learning_rate * gradient
            loss = np.mean(errors**2)         # теперь верный loss

            if verbose and i % verbose == 0:
                print(f"iter {i} | loss: {loss}")
                print(self.weights)


    def get_coef(self):
        return self.weights[1:]
    
    def predict(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)

        y_pred = X_with_bias @ self.weights
        
        return sum(y_pred)




X = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
y = pd.Series([0, 1])

verbose = 10
MyLineReg1 = MyLineReg()
print(MyLineReg1)
MyLineReg1.fit(X, y, verbose)