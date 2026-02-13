import pandas as pd
import numpy as np


class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)

qqq = MyLogReg()

print(qqq)