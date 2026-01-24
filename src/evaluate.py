from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)



def evaluate(model, X_test, y_test, threshold=0.5, show_cm=True):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    print(f"\n✅ Evaluation (threshold = {threshold:.2f})")
    print("Accuracy:", accuracy_score(y_test, preds))

    if show_cm:
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    print(classification_report(y_test, preds))
    return proba, preds


def find_best_threshold(model, X_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    proba = model.predict_proba(X_test)[:, 1]

    best_t = None
    best_acc = -1

    print("\n🔍 Scanning thresholds...")
    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"threshold={t:.2f}  accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"\n🔥 Best threshold: {best_t:.2f} (acc={best_acc:.4f})")
    return best_t
