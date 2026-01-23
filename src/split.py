import pandas as pd
def time_split(df,target_col,test_size=0.2):
    df=df.sort_index()
    
    split_idx=int(len(df)*(1-test_size))
    #point at idx where split need to happen
    X=df.drop(columns=[target_col])
    y=df[target_col]
    
    X_train=X.iloc[:split_idx]
    X_test=X.iloc[split_idx:]
    y_train=y.iloc[:split_idx]
    y_test=y.iloc[split_idx:]
    
    print("Total:", len(df))
    print("Train:", len(X_train))
    print("Test :", len(X_test))

    return X_train,X_test,y_train,y_test