from sklearn.ensemble import RandomForestClassifier

def train_model(X_train,y_train):
    model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
    
    model.fit(X_train,y_train)

    return model