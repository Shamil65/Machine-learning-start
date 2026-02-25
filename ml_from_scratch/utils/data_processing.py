import pandas as pd
import numpy as np
import random

def _train_test_split(X, y, train_size=0.8, random_state=3310, as_frame=False):
    # X, y - это датафреймы (dataframe)

    np.random.seed(random_state)
    full_data = X.copy()
    full_data["target"] = y
    full_data_np = full_data.values

    np.random.shuffle(full_data_np)

    n = full_data_np.shape[0]
    count_of_array = int(n * train_size)

    train_data = full_data_np[:count_of_array]
    test_data = full_data_np[count_of_array:]

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]


    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    if as_frame:
        return pd.DataFrame(X_train), pd.Series(y_train), pd.DataFrame(X_test), pd.Series(y_test)
    else:
        return X_train, y_train, X_test, y_test