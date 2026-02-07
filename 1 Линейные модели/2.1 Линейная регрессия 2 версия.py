import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.001, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return "MyLineReg class: n_iter={}, learning_rate={}".format(self.n_iter, self.learning_rate)

    def fit(self, X, y, verbose=False):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        m, n = X_with_bias.shape # m - кол-во строк; n - кол-во столбцов

        self.weights = np.ones(n) # засовывваем в эту переменную np массив со значениями 1 в кол-ве равном кол-ву переменных в одном векторе в X
        
        X_mat = X_with_bias.values # Превращаем из DataFrame в np array
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
    
    def mae(y_true, y_pred):
        errors = abs(y_pred - y_true)
        return np.mean(errors)

    def mse(y_true, y_pred):
        errors = y_pred - y_true
        return np.mean(errors**2)


    def rmse(y_true, y_pred):
        pass

    def mape(y_true, y_pred):
        pass

    def r2(y_true, y_pred):
        pass
        
    




X = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
y = pd.Series([0, 1])

X_test = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})

verbose = 10
MyLineReg1 = MyLineReg()
print(MyLineReg1)
MyLineReg1.fit(X, y, verbose)
print(MyLineReg1.predict(X_test))

MyLineReg1.metric(mse=1)
