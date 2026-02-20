import random
import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.001, weights=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state  = random_state


    def __str__(self):
        return "MyLineReg class: n_iter={}, learning_rate={}".format(self.n_iter, self.learning_rate)
    
    def sign_(self):
        return np.sign(self.weights)


    def fit(self, X, y, verbose=False):

        random.seed(self.random_state)
    
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        m, n = X_with_bias.shape
        self.weights = np.ones(n)
        X_mat = X_with_bias.values # Превращаем из DataFrame в np array
        y_vec = y.values
        y_pred = X_mat @ self.weights

        if verbose and self.reg == "l1":
            MSE = np.mean((y_pred - y_vec)**2)
            L1_loss = MSE + self.l1_coef * np.sum(np.abs(self.weights))
            print(f"start | loss: {L1_loss}")

        elif verbose and self.reg == "l2":
            MSE = np.mean((y_pred - y_vec)**2)
            L2_loss = MSE + self.l2_coef * np.sum((self.weights)**2)
            print(f"start | loss: {L2_loss}")

        elif verbose and self.reg == "elasticnet":
            MSE = np.mean((y_pred - y_vec)**2)
            ElasticNet = MSE + ( self.l1_coef * np.sum(np.abs(self.weights)) ) + ( self.l2_coef * np.sum((self.weights)**2) )
            print(f"start | loss: {ElasticNet}")  

        elif verbose and self.reg == None:
            MSE = np.mean((y_pred - y_vec)**2)
            print(f"start | loss: {MSE}")  
            

        is_lr_callable = callable(self.learning_rate)

        for i in range(1, self.n_iter + 1):

            n_count = X_with_bias.shape[0]

            if type(self.sgd_sample) == int:
                batch_size = self.sgd_sample
            elif type(self.sgd_sample) == float:
                batch_size = round(self.sgd_sample * n_count)
            else:
                batch_size = n_count

            sample_rows_idx = random.sample(range(X_with_bias.shape[0]), batch_size)
            X_batch_df = X_with_bias.iloc[sample_rows_idx]
            y_batch = y.iloc[sample_rows_idx].values  # ← КЛЮЧЕВАЯ СТРОКА!

            X_mat = X_batch_df.values
            m = X_mat.shape[0] 


            if is_lr_callable:
                self.learning_rate_iter = self.learning_rate(i)
            else:
                self.learning_rate_iter = self.learning_rate

            y_pred = X_mat @ self.weights
            errors = y_pred - y_batch
            gradient = (2 / m) * (X_mat.T @ errors)

            

            if self.reg == "l1":
                L1_gradient = gradient + self.l1_coef * np.sign(self.weights)
                self.weights = self.weights - self.learning_rate_iter * L1_gradient

                if verbose and i % verbose == 0:
                    L1_loss = MSE + ( self.l1_coef * np.sum(np.abs(self.weights)) ) + ( self.l2_coef * np.sum((self.weights)**2) )
                    print(f"iter {i} | loss: {L1_loss}")

                
            elif self.reg == "l2":
                L2_gradient = gradient + self.l2_coef * 2 * self.weights
                self.weights = self.weights - self.learning_rate_iter * L2_gradient

                if verbose and i % verbose == 0:
                    L2_loss = MSE + self.l2_coef * np.sum((self.weights)**2)
                    print(f"iter {i} | loss: {L2_loss}")

            
            elif self.reg == "elasticnet":
                ElasticNet_gradient = gradient + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
                self.weights = self.weights - self.learning_rate_iter * ElasticNet_gradient

                if verbose and i % verbose == 0:
                    ElasticNet_loss = MSE + ( self.l1_coef * np.sum(np.abs(self.weights)) ) + ( self.l2_coef * np.sum((self.weights)**2) )
                    print(f"iter {i} | loss: {ElasticNet_loss}")

            elif self.reg == None:
                self.weights = self.weights - self.learning_rate_iter * gradient

                if verbose and i % verbose == 0:
                    MSE = np.mean((y_pred - y_batch)**2)
                    print(f"iter {i} | loss: {MSE}")

    def get_coef(self):
        return np.mean(self.weights[1:])
    
    def predict(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)

        y_pred = X_with_bias.values @ self.weights
        return y_pred