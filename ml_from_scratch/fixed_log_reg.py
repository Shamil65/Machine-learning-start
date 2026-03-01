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


    def accuracy(self, X, y, weights):
        y_pred = (self.sigmoid(X @ self.weights) > 0.5).astype(int)
        return np.mean(y_pred == y)
    

    def precision(self, X, y):
        y_pred = (self.sigmoid(X @ self.weights) > 0.5).astype(int)

        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))

        return TP / (TP + FP + 1e-15)


    def recall(self, X, y):
        y_pred = (self.sigmoid(X @ self.weights) > 0.5).astype(int)

        TP = np.sum((y_pred == 1) & (y == 1))
        FN = np.sum((y_pred == 0) & (y == 1))

        return TP / (TP + FN + 1e-15)


    def f1(self, X, y):
        p = self.precision(X, y)
        r = self.recall(X, y)

        return 2 * p * r / (p + r + 1e-15)


    def roc_auc(self, X, y):
        # X — матрица с bias, y — метки классов (0/1)
        y_score = self.sigmoid(X @ self.weights)  # предсказанные вероятности
        q = len(y)
        
        P = np.sum(y == 1)  # число положительных
        N = np.sum(y == 0)  # число отрицательных
        
        auc_sum = 0.0
        
        for i in range(q):
            for j in range(q):
                # Проверка y_i < y_j
                I_y = 1 if y[i] < y[j] else 0
                
                # Проверка a_i < a_j
                if y_score[i] < y_score[j]:
                    I_a = 1
                elif y_score[i] == y_score[j]:
                    I_a = 0.5
                else:
                    I_a = 0
                
                auc_sum += I_y * I_a
        
        auc = auc_sum / (P * N + 1e-15)  # делим на P*N для нормализации
        return auc


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

        print(self.accuracy(X_mat, y_vec, self.weights))
            

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