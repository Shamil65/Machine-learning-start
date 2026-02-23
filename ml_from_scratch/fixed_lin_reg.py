import random
import pandas as pd
import numpy as np


class My_Line_Reg():
    # Класс с линейной регрессией

    def __init__(self, n_iter=100, learning_rate=0.001, weights=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        # Функция для инициализации
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state  = random_state

        self.configuration_regularization = {
            "l1": 
                    {"l1_coef": self.l1_coef,
                    "l2_coef": 0},
            "l2": 
                    {"l1_coef": 0,
                    "l2_coef": self.l2_coef},
            "elasticnet": 
                    {"l1_coef": self.l1_coef,
                    "l2_coef": self.l2_coef},
            "none": 
                    {"l1_coef": 0,
                    "l2_coef": 0}
        }


    def __str__(self):
        # Функция с выводом информации о классе
        return "MyLineReg class: n_iter={}, learning_rate={}".format(self.n_iter, self.learning_rate)
    

    def _calculate_loss(self, y_pred, y_vec, config):
        # Функция для рассчета loss
        MSE = np.mean((y_pred - y_vec)**2)
        penalty_l1 = config["l1_coef"] * np.sum(np.abs(self.weights))
        penalty_l2 = config["l2_coef"] * np.sum((self.weights)**2)
        loss = MSE + penalty_l1 + penalty_l2
        return loss
        

    def _prepare_data(self, X, y):
        # Функция для подготовки данных
        ones_np_array = np.ones(len(X))
        ones_df = pd.DataFrame(ones_np_array, columns=["bias"], index=X.index)
        full_X = pd.concat([ones_df, X], axis=1)
        y_vec = y.values
        full_X_mat = full_X.values
        return full_X_mat, y_vec

    
    def _log(self, iteration, loss):
        if iteration == 1:
            print(f"start | loss: {loss}")
        else:
            print(f"iter {iteration} | loss: {loss}")


    def fit(self,  X, y):
        # Функция с обучением
        random.seed(self.random_state)
        X_mat, y_vec = self._prepare_data(X, y)
        n_features = X_mat.shape[1]
        self.weights = np.ones(n_features)
        
        # = reg если оно не равно None, иначе оно равно none
        key = self.reg if self.reg is not None else "none"

        config = self.configuration_regularization[key]
        y_pred = X_mat @ self.weights
        loss = self._calculate_loss(y_pred, y_vec, config)
        print(loss)
        



X = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
y = pd.Series([0, 1])

X_test = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})

MyLineReg1 = My_Line_Reg(reg="elasticnet", l2_coef=2)
print(MyLineReg1)
MyLineReg1.fit(X, y)


# [Псевдокод на естественном языке]

# метод fit(X, y, verbose=False):
#     → подготовить данные
#     → инициализировать веса
#     → нормализовать ключ регуляризации
#     → получить конфигурацию
#     → если verbose:
#         → сделать прогноз на полных данных
#         → вызвать _calculate_loss(прогноз, y, конфигурация)
#         → вызвать _log("start", loss)
#     → цикл обучения

# метод _calculate_loss(y_pred, y_vec, config):
#     → посчитать MSE
#     → посчитать штраф L1 через config.l1_coef
#     → посчитать штраф L2 через config.l2_coef
#     → вернуть сумму

# метод _log(iteration, loss):
#     → форматированный вывод