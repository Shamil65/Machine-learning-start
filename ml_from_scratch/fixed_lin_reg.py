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
    

    def _calculate_loss(self, y_pred, y_vec):
        # Функция для рассчета loss
        MSE = np.mean((y_pred - y_vec)**2)
        penalty_l1 = self.l1_coef * np.sum(np.abs(self.weights))
        penalty_l2 = self.l2_coef * np.sum((self.weights)**2)
        loss = MSE + penalty_l1 + penalty_l2
        return loss
        

    def _prepare_data(self, X, y):
        # Функция для подготовки данных
        ones_np_array = np.ones(len(X))
        print(ones_np_array)
        ones_df = pd.DataFrame(ones_np_array, columns=["bias"])
        full_X = pd.concat([ones_df, X], axis=1)
        y_vec = y.values
        return full_X, y_vec


    def fit(self,  X, y):
        # Функция с обучением

        random.seed(self.random_state)

        print(self.configuration_regularization[self.reg])
        X_mat, y_vec = self._prepare_data
        print(X_mat)
        print(y_vec)



# [Псевдокод на естественном языке]

# метод fit:
#     → вызвать _initialize(X, y)
#     → если verbose: вывести начальный loss
#     → цикл обучения:
#         → взять батч
#         → посчитать градиент
#         → обновить веса
#         → если verbose и нужная итерация:
#             → посчитать loss на полной выборке
#             → вывести через _log

# метод _calculate_loss(X, y, weights):
#     → посчитать MSE
#     → если L1: добавить штраф L1
#     → если L2: добавить штраф L2
#     → если ElasticNet: добавить оба штрафа
#     → вернуть итоговое значение

# метод _log(iteration, loss):
#     → форматированный вывод