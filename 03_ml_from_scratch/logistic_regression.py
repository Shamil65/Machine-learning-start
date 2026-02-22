import pandas as pd
import numpy as np


class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def sigmooid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def loss_function(self, X, y, weight):
        m = len(y)
        eps = 1e-15

        y_pred = self.sigmooid(X @ weight)

        loss = (-1/m) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

        return loss
    

    def fit(self, X, y, verbose=False):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        m, n = X_with_bias.shape
        self.weights = np.ones(n)

        X_mat = X_with_bias.values # Превращаем из DataFrame в np array
        y_vec = y.values

        # Расчет y_pred

        y_pred = X_with_bias @ self.weights
        start_loss = self.loss_function(X_mat, y_vec, self.weights)

        if verbose:
            print(f"start | loss: {start_loss:.2f}")

        for i in range(1, self.n_iter + 1):

            y_pred = self.sigmooid(X_mat @ self.weights)

            gradient = (X_mat.T @ (y_pred - y_vec))/m

            self.weights -= self.learning_rate * gradient
            
            if verbose and i % verbose == 0:
                current_loss = self.loss_function(X_mat, y_vec, self.weights)
                print(f"{i} | loss: {current_loss:.2f}")

    def get_coef(self):
        return self.weights[1:]
    
    def predict(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        X_math = X_with_bias.values

        y_pred = self.sigmooid(X_math @ self.weights)
        y_pred_bin = y_pred > 0.5
        return y_pred_bin


    def predict_proba(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        X_math = X_with_bias.values

        y_pred = self.sigmooid(X_math @ self.weights)

        return y_pred
        



            # y_pred = X_with_bias @ self.weights
            # Log_loss = (-1/n) * np.sum(y_vec*np.log(y_pred) + (1+y_vec)*np.log(1-y_pred))





X = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
y = pd.Series([0, 1])

X_test = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})

verbose = 10
metric="mse"

MyLineReg1 = MyLogReg(n_iter=100, learning_rate=0.1)
print(MyLineReg1)

MyLineReg1.fit(X, y, verbose)
# print(MyLineReg1.predict(X_test))

# MyLineReg1.get_best_score()