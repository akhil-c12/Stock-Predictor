from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
def find_best_threshold(model, X_val, y_val, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    # Use scaler if available
    X_val_scaled = model.scaler.transform(X_val) if hasattr(model, 'scaler') else X_val
    proba = model.predict_proba(X_val_scaled)[:, 1]

    best_t = None
    best_acc = -1

    print("\n🔍 Scanning thresholds on VALIDATION set...")
    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        print(f"threshold={t:.2f}  accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"\n🔥 Best threshold (VAL): {best_t:.2f} (acc={best_acc:.4f})")
    return best_t


def evaluate(model, X_test, y_test, threshold=0.5, show_cm=True):
    # Use scaler if available
    X_test_scaled = model.scaler.transform(X_test) if hasattr(model, 'scaler') else X_test
    proba = model.predict_proba(X_test_scaled)[:, 1]
    preds = (proba >= threshold).astype(int)

    print(f"\n✅ Evaluation on TEST (threshold = {threshold:.2f})")
    print("Accuracy:", accuracy_score(y_test, preds))

    if show_cm:
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    print(classification_report(y_test, preds))
    return proba, preds
def add_model_proba(df, model, threshold):
    X = df.drop(columns=["target"])
    X_scaled = model.scaler.transform(X)
    df["proba"] = model.predict_proba(X_scaled)[:, 1]
    df["signal"] = np.where(
        df["proba"] >= threshold, 1,
        np.where(df["proba"] <= 0.35, -1, 0)
    )
    return df
