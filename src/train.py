from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_model(X_train, y_train, X_val, y_val, model_params: dict):
    """Trains a Random Forest model with optimized hyperparameters and feature scaling."""
    
    # Feature scaling - improves model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Calculate class weights to handle slight imbalance
    class_weights = {0: 1.0 / (y_train == 0).sum(), 1: 1.0 / (y_train == 1).sum()}
    class_weight_balanced = {0: class_weights[0] / min(class_weights.values()),
                             1: class_weights[1] / min(class_weights.values())}
    
    # Optimized Random Forest parameters
    rf_params = {
        "n_estimators": model_params.get("n_estimators", 300),  # More trees for stability
        "max_depth": model_params.get("max_depth", 12),  # Reduced depth to reduce overfitting
        "min_samples_split": model_params.get("min_samples_split", 8),  # Higher to reduce overfitting
        "min_samples_leaf": model_params.get("min_samples_leaf", 4),  # Higher to reduce overfitting
        "max_features": model_params.get("max_features", "sqrt"),  # sqrt for classification
        "class_weight": "balanced",  # Handle class imbalance
        "random_state": model_params.get("random_state", 42),
        "n_jobs": model_params.get("n_jobs", -1),
    }
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_scaled, y_train)
    
    # Store scaler for later use
    model.scaler = scaler
    
    return model
