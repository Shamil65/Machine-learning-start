import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.001, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.loss_array = []

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
        loss_mse = np.mean((y_pred - y_vec)**2)
        
        # Необходимо внутри цикла реализовать выбор метрики и показ, 
        # но само обучение будет проиходить с помощью mse

        if verbose:
            if self.metric == "mae": # start | loss: 42027.65 | <metric_name>: 234.65
                print(f"start | loss: {loss_mse} | mae: {self.mae(y_vec, y_pred)}")
            elif self.metric == "mse":
                print(f"start | loss: {loss_mse} | mse: {self.mse(y_vec, y_pred)}")
            elif self.metric == "rmse":
                print(f"start | loss: {loss_mse} | rmse: {self.rmse(y_vec, y_pred)}")
            elif self.metric == "mape":
                print(f"start | loss: {loss_mse} | mape: {self.mape(y_vec, y_pred)}")
            elif self.metric == "r2":
                print(f"start | loss: {loss_mse} | r2: {self.r2(y_vec, y_pred)}")
            elif self.metric == None:
                print(f"start | loss: {loss_mse}")

       

        for i in range(1, self.n_iter + 1):
            y_pred = X_mat @ self.weights
            errors = y_pred - y_vec         
            gradient = (2 / m) * (X_mat.T @ errors)
            self.weights = self.weights - self.learning_rate * gradient
            loss_mse = np.mean((y_pred - y_vec)**2)


            # if verbose and i % verbose == 0:
            #     print(f"iter {i} | loss: {loss}")
            #     print(self.weights)

            if verbose and i % verbose == 0:   # 100 | loss: 1222.87 | <metric_name>: 114.35
                if self.metric == "mae":
                    print(f"{i} | loss: {loss_mse} | mae: {self.mae(y_vec, y_pred)}")
                elif self.metric == "mse":
                    print(f"{i} | loss: {loss_mse} | mse: {self.mse(y_vec, y_pred)}")
                elif self.metric == "rmse":
                    print(f"{i} | loss: {loss_mse} | rmse: {self.rmse(y_vec, y_pred)}")
                elif self.metric == "mape":
                    print(f"{i} | loss: {loss_mse} | mape: {self.mape(y_vec, y_pred)}")
                elif self.metric == "r2":
                    print(f"{i} | loss: {loss_mse} | r2: {self.r2(y_vec, y_pred)}")
                elif self.metric == None:
                    print(f"{i} | loss: {loss_mse}")

        y_pred = X_mat @ self.weights
        
        self.loss_array.append(self.mae(y_vec, y_pred))
        self.loss_array.append(self.mse(y_vec, y_pred))
        self.loss_array.append(self.rmse(y_vec, y_pred))
        self.loss_array.append(self.mape(y_vec, y_pred))
        self.loss_array.append(self.r2(y_vec, y_pred))

    def get_best_score(self):
        if self.metric == "mae":
            return self.loss_array[0]
        elif self.metric == "mse":
            return self.loss_array[1]
        elif self.metric == "rmse":
            return self.loss_array[2]
        elif self.metric == "mape":
            return self.loss_array[3]
        elif self.metric == "r2":
            return self.loss_array[4]
        elif self.metric == None:
            pass


    def get_coef(self):
        return self.weights[1:]
    
    def predict(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        return X_with_bias.values @ self.weights
        
    
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
        part_of_mape = np.sum(np.abs((y_pred - y_true)/(y_true)))
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
metric="mse"

MyLineReg1 = MyLineReg(n_iter=100, learning_rate=0.1, metric="rmse")
print(MyLineReg1)

MyLineReg1.fit(X, y, verbose)
print(MyLineReg1.predict(X_test))

MyLineReg1.get_best_score()
