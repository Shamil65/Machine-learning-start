import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.001, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric

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
        
        # Необходимо внутри цикла реализовать выбор метрики и показ, 
        # но само обучение будет проиходить с помощью mse

        if self.metric == "mae" and verbose:
            print(f"start | loss: {self.mae(y_vec, y_pred)}")
        elif self.metric == "mse" and verbose:
            print(f"start | loss: {self.mse(y_vec, y_pred)}")
        elif self.metric == "rmse" and verbose:
            print(f"start | loss: {self.rmse(y_vec, y_pred)}")
        elif self.metric == "mape" and verbose:
            print(f"start | loss: {self.mape(y_vec, y_pred)}")
        elif self.metric == "r2" and verbose:
            print(f"start | loss: {self.r2(y_vec, y_pred)}")

        for i in range(1, self.n_iter + 1):
            y_pred = X_mat @ self.weights
            errors = y_pred - y_vec         
            gradient = (2 / m) * (X_mat.T @ errors)
            self.weights = self.weights - self.learning_rate * gradient
            loss = np.mean(errors**2)         # теперь верный loss

            if verbose and i % verbose == 0:
                print(f"iter {i} | loss: {loss}")
                print(self.weights)

            if verbose and i % verbose == 0:   
                if self.metric == "mae":
                    print(f"iter {i} | mae: {self.mae(y_vec, y_pred)}")
                elif self.metric == "mse":
                    print(f"iter {i} | mse: {self.mse(y_vec, y_pred)}")
                elif self.metric == "rmse":
                    print(f"iter {i} | rmse: {self.rmse(y_vec, y_pred)}")
                elif self.metric == "mape":
                    print(f"iter {i} | mape: {self.mape(y_vec, y_pred)}")
                elif self.metric == "r2":
                    print(f"iter {i} | r2: {self.r2(y_vec, y_pred)}")


    def get_coef(self):
        return self.weights[1:]
    
    def predict(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        y_pred = X_with_bias @ self.weights
        
        return sum(y_pred)
    
    def mae(self, y_true, y_pred):
        errors = abs(y_pred - y_true)
        return np.mean(errors)

    def mse(self, y_true, y_pred):
        errors = y_pred - y_true
        return np.mean(errors**2)


    def rmse(self, y_true, y_pred):
        errors = y_pred - y_true
        mse = np.mean(errors**2)
        return np.sqrt(mse)

    def mape(self, y_true, y_pred):
        part_of_mape = np.sum(np.abs((y_pred - y_true)/(y_pred)))
        n = len(y_true)
        return (100/n) * part_of_mape

    def r2(self, y_true, y_pred):
        errors = y_pred - y_true
        numerator_r2 = np.sum(errors**2)
        denominator_r2 = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (numerator_r2 / denominator_r2)
        

        
    




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
