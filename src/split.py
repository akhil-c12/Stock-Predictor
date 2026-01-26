import pandas as pd

def time_split_3way(df, target_col="target", train_size=0.70, val_size=0.15):

    df = df.copy()
    df = df.sort_index()

    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val   = y.iloc[train_end:val_end]
    y_test  = y.iloc[val_end:]

    print("Total:", n)
    print("Train:", len(X_train))
    print("Val  :", len(X_val))
    print("Test :", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test
