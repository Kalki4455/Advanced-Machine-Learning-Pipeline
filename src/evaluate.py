from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ModelEvaluation:

    def evaluate(self, y_test, predictions):

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            average="weighted"
        )
        recall = recall_score(
            y_test,
            predictions,
            average="weighted"
        )
        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, predictions))