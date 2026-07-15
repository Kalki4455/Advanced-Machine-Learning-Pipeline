import os
import joblib
from sklearn.ensemble import RandomForestClassifier


class IrisModel:
    """
    Random Forest Model Class
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def train(self, X_train, y_train):
        """
        Train the model
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using trained model
        """
        return self.model.predict(X_test)

    def feature_importance(self, feature_names):
        """
        Display Feature Importance
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)

        importance = self.model.feature_importances_

        for feature, score in zip(feature_names, importance):
            print(f"{feature:<20} : {score:.4f}")

    def save(self):
        """
        Save trained model
        """
        os.makedirs("models", exist_ok=True)

        model_path = os.path.join("models", "iris_model.pkl")

        joblib.dump(self.model, model_path)

        print(f"\n✅ Model Saved Successfully")
        print(f"Location : {model_path}")

    def load(self):
        """
        Load saved model
        """
        model_path = os.path.join("models", "iris_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found.")

        self.model = joblib.load(model_path)

        print("✅ Model Loaded Successfully")