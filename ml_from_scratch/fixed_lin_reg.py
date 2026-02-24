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
        # Функция для логирование итераций
        if iteration == 0:
            print(f"start | loss: {loss}")
        else:
            print(f"iter {iteration} | loss: {loss}")

    
    def _get_batch(self, X, y, sgd_sample):
        # Функция для разделения данных для стахостического градиентного спуска
        n_count = X.shape[0]

        if type(sgd_sample) == int:
            batch_size = sgd_sample
        elif type(sgd_sample) == float:
            batch_size = round(sgd_sample * n_count)
        else:
            batch_size = n_count
        
        sample_rows_idx = random.sample(range(n_count), batch_size)
        X_batch = X[sample_rows_idx]
        y_batch = y[sample_rows_idx]
        
        return X_batch, y_batch
    

    def fit(self,  X, y, verbose=False):
        # Функция с обучением
        random.seed(self.random_state)
        X_mat, y_vec = self._prepare_data(X, y)
        n_features = X_mat.shape[1]
        self.weights = np.ones(n_features)
        
        # = reg если оно не равно None, иначе оно равно none
        key = self.reg if self.reg is not None else "none"

        config = self.configuration_regularization[key]
        y_pred = X_mat @ self.weights
        
        
        for i in range(0, self.n_iter + 1):
            # цикл с обучением

            if verbose and i % verbose == 0:
                y_pred_full_data = X_mat @ self.weights
                loss_ = self._calculate_loss(y_pred_full_data, y_vec, config)
                self._log(i, loss_)

            # специально добавили один цикл для вывод старт start
            # поэтому сделали 0, self.n_iter + 1
            if i == 0:
                continue

            X_batch, y_batch = self._get_batch(X_mat, y_vec, self.sgd_sample)

            y_pred = X_batch @ self.weights
            errors = y_pred - y_batch

            m = X_batch.shape[0] 
            gradient = (2 / m) * (X_batch.T @ errors)

            regularized_gradient = gradient + config["l1_coef"] * np.sign(self.weights) + config["l2_coef"] * 2 * self.weights
            self.weights -= regularized_gradient * self.learning_rate
        
        y_pred_final = X_mat @ self.weights
        print(self._r2(y_vec, y_pred_final))

            
    def _r2(self, y_vec, y_pred):
        residual_sum_of_squares = np.sum((y_vec - y_pred)**2) 
        total_sum_of_squares = np.sum((y_vec - np.mean(y_vec))**2) 

        return 1 - (residual_sum_of_squares/total_sum_of_squares)


X = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
y = pd.Series([0, 1])

X_test = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
verbose = 1000
MyLineReg1 = My_Line_Reg(reg="elasticnet", l2_coef=2, n_iter=10000)
print(MyLineReg1)
MyLineReg1.fit(X, y, verbose)
