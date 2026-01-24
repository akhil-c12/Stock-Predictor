from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, model_params: dict):
    """Trains the model."""
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model
