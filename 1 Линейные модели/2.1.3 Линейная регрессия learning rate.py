import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.001, weights=None, reg=None, l1_coef=0.0, l2_coef=0.0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return "MyLineReg class: n_iter={}, learning_rate={}".format(self.n_iter, self.learning_rate)
    
    def sign_(self):
        return np.sign(self.weights)


    def fit(self, X, y, verbose=False):
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

            if is_lr_callable:
                self.learning_rate_iter = self.learning_rate(i)
            else:
                self.learning_rate_iter = self.learning_rate

            y_pred = X_mat @ self.weights
            errors = y_pred - y_vec
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
                    MSE = np.mean((y_pred - y_vec)**2)
                    print(f"iter {i} | loss: {MSE}")
            

                





             


    # def fit(self, X, y, verbose=False):
    #     X_with_bias = X.copy()
    #     X_with_bias.insert(0, "bias", 1)
    #     m, n = X_with_bias.shape # m - кол-во строк; n - кол-во столбцов

    #     self.weights = np.ones(n) # засовывваем в эту переменную np массив со значениями 1 в кол-ве равном кол-ву переменных в одном векторе в X
    #     print("self.weights", self.weights)

    #     X_mat = X_with_bias.values # Превращаем из DataFrame в np array
    #     y_vec = y.values
    #     y_pred = X_mat @ self.weights

    #     errors = y_pred - y_vec
    #     MSE = np.mean(errors**2)
    #     L1_penalty = np.sign(self.weights)
    #     L1_gradient = gradient + self.l1_coef * L1_penalty



    #     if verbose and self.reg=="l1":
    #         #mse_l1 = MSE + l1_coef * 
    #         print(f"start | loss: {MSE}")

    #     for i in range(1, self.n_iter + 1):
    #         y_pred = X_mat @ self.weights
    #         errors = y_pred - y_vec           # 🔹 ОБЯЗАТЕЛЬНО обновляем
    #         gradient = (2 / m) * (X_mat.T @ errors)
    #         # Надо к этому градиенту прибавить массив, котрый по размеру должен совпадать с тем что будет в исходном, то 
    #         # есть по кол-ву равен кол-ву весов. Это значит надо сделать функцию, куда подаются веса и приходят штрафы, можно это сделать через 
    #         # self чтобы отдельно не подавать даные в функцию, а промто через self подавать инфу 
    #         L1_penalty = np.sign(self.weights)
    #         L1_gradient = gradient + self.l1_coef * L1_penalty
            
    #         self.weights = self.weights - self.learning_rate * L1_gradient
    #         loss = np.mean(errors**2)         # теперь верный loss

    #         if verbose and i % verbose == 0:
    #             print(f"iter {i} | loss: {loss}")
    #             print(self.weights)


    def get_coef(self):
        return self.weights[1:]
    
    def predict(self, X):
        X_with_bias = X.copy()
        X_with_bias.insert(0, "bias", 1)

        y_pred = X_with_bias @ self.weights
        
        return y_pred




X = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})
y = pd.Series([0, 1])

X_test = pd.DataFrame({"x1": [3, 2],
                  "x2": [6, 4],
                  "x3": [9, 8]})

verbose = 10
MyLineReg1 = MyLineReg()
print(MyLineReg1)
MyLineReg1.fit(X, y, verbose)
print(MyLineReg1.predict(X_test))
print(MyLineReg1.sign_())