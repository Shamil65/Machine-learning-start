import pandas as pd
import numpy as np


class MyLogReg():
    # Класс для реализации логистической регрессии
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights


    def __str__(self):
        # Функция для вывода информации о модели
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"


    def sigmoid(self, z):
        # Функция для вычисления сигмоида
        return 1.0 / (1.0 + np.exp(-z))
    
    
    def _calculate_loss(self, X, y, weights):
        # Функция для расчета функции потерь
        m = len(y)
        eps = 1e-15
        y_pred = self.sigmoid(X @ weights)
        loss = (-1/m) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        return loss
    

    def _prepare_data(self, X, y):
        # Функция для подготовки данных
        ones_np_array = np.ones(len(X))
        ones_df = pd.DataFrame(ones_np_array, columns=["bias"], index=X.index)
        full_X = pd.concat([ones_df, X], axis=1)
        full_X_mat = full_X.values
        y_vec = y.values
        return full_X_mat, y_vec
    

    def _log(self, i, loss):
        # Функция для вывода информации
        if i == 0:
            print(f"start | loss: {loss:.2f}")
        else:
            print(f"{i} | loss: {loss:.2f}")


    def fit(self, X, y, verbose=False):
        # Функция для обучения модели
        X_mat, y_vec = self._prepare_data(X, y)
        m_instances, n_features = X_mat.shape
        self.weights = np.ones(n_features)

        for i in range(0, self.n_iter + 1):
            
            if verbose and i % verbose == 0:
                loss = self._calculate_loss(X_mat, y_vec, self.weights)
                self._log(i, loss)
                
            if i == 0:
                continue

            y_pred = self.sigmoid(X_mat @ self.weights)
            gradient = (X_mat.T @ (y_pred - y_vec))/m_instances
            self.weights -= self.learning_rate * gradient
            

    def get_coef(self):
        # Функция для вывода коэффициентов
        return self.weights[1:]
    
    def predict(self, X):
        # predict – переводит вероятности в бинарные классы по порогу > 0.5
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        X_math = X_with_bias.values

        y_pred = self.sigmoid(X_math @ self.weights)
        y_pred_bin = y_pred > 0.5
        return y_pred_bin


    def predict_proba(self, X):
        # predict_proba – возвращает вероятности (логиты прогнанные через функцию сигмоиды)
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        X_math = X_with_bias.values

        y_pred = self.sigmoid(X_math @ self.weights)

        return y_pred






# X = pd.DataFrame({"x1": [3, 2],
#                   "x2": [6, 4],
#                   "x3": [9, 8]})
# y = pd.Series([0, 1])

# X_test = pd.DataFrame({"x1": [3, 2],
#                   "x2": [6, 4],
#                   "x3": [9, 8]})

# verbose = 10
# metric="mse"

# MyLineReg1 = MyLogReg(n_iter=100, learning_rate=0.1)
# print(MyLineReg1)

# MyLineReg1.fit(X, y, verbose)
# # print(MyLineReg1.predict(X_test))

# # MyLineReg1.get_best_score()