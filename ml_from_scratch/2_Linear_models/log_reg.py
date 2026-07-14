import pandas as pd
import numpy as np


class MyLogReg():
    # Класс для реализации логистической регрессии
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None


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
        y_true = y_vec

        for i in range(0, self.n_iter + 1):
            
            if verbose and i % verbose == 0:
                loss = self._calculate_loss(X_mat, y_vec, self.weights)
                self._log(i, loss)
                
            if i == 0:
                continue

            y_pred = self.sigmoid(X_mat @ self.weights)
            gradient = (X_mat.T @ (y_pred - y_vec))/m_instances
            self.weights -= self.learning_rate * gradient

        
    def _calculate_metric(self, y_true, y_pred_proba):
        # Если метрика не задана — выходим
        if self.metric == None:
            return
        

        # y_true = [0, 1, 0, 0, 1, 0, 1, 1, 0]  
        # y_pred = [0, 0, 0, 0, 1, 0, 1, 1, 0]
        # 
        # TP (True Positive) – истинно положительные примеры;
        # TN (True Negative) – истинно отрицательные примеры;
        # FP (False Positive) – ложноположительные примеры;
        # FN (False Negative) – ложноотрицательные примеры.
        # 
        # TP - 3 | FP - 0
        # FN - 1 | TN - 5


        # Получаем бинарные предсказания по порогу 0.5
        TP = np.sum((y_true == 1) & (y_pred_proba == 1))
        TN = np.sum((y_true == 0) & (y_pred_proba == 0))
        FP = np.sum((y_true == 0) & (y_pred_proba == 1))
        FN = np.sum((y_true == 1) & (y_pred_proba == 0))


        print("TP: ", TN, "/nTN: ", TP, "/nFP: ", FP, "/nFN: ", FN)

        # Расчет метрик accuracy, precision, recall, f1
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    

        if self.metric == 'accuracy':
            return accuracy
        
        if self.metric == 'precision':
            return precision
        
        if self.metric == 'recall':
            return recall
        
        if self.metric == 'f1':
            return f1

        
        if self.metric == 'roc_auc':
            return 
        


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