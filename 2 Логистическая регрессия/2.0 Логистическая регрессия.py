import pandas as pd
import numpy as np


class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)
        m, n = X_with_bias.shape
        self.weights = np.ones(n)

        # Расчет y_pred
        y_pred = X_with_bias @ self.weights

        print(y_pred)


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