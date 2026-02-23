import os

from ml_from_scratch import linear_regression

# C:\Users\ZiganshinShamil\Desktop\Study\ml_study\03_ml_from_scratch\linear_regression.py


# from linear_regression import MyLineReg

model_linear = linear_regression.MyLineReg(n_iter=10000, sgd_sample=0.2, learning_rate=0.01)

model_linear.fit(X, y, verbose = 1000)
print(model_linear)


