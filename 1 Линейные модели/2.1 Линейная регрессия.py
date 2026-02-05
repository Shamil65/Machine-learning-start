import numpy as np
import pandas as pd


class MyLineReg:
    def init(self, n_iter=100, learning_rate=0.01, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def fit(self, X, y, verbose=False):
        X_ = X.copy()
        X_.insert(0, 'bias', 1)

        X_mat = X_.values
        y_vec = y.values

        m, n = X_mat.shape

        self.weights = np.ones(n)

        y_pred = X_mat @ self.weights
        loss = np.mean((y_pred - y_vec)**2)
        if verbose:
            print(f"start | loss: {loss}")

        for i in range(1, self.n_iter + 1):
            y_pred = X_mat @ self.weights
            errors = y_pred - y_vec
            gradient = (2 / m) * (X_mat.T @ errors)
            self.weights -= self.learning_rate * gradient

            loss = np.mean(errors**2)
            if verbose and i % verbose == 0:
                print(f"{i} | loss: {loss}")

    def get_coef(self):
        return self.weights[1:]