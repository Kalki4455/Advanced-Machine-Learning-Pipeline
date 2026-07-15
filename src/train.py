import os

from data_loader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineering
from model import IrisModel
from evaluate import ModelEvaluation


def main():

    print("=" * 60)
    print("IRIS MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # Load Dataset
    loader = DataLoader(os.path.join("data", "iris.csv"))
    df = loader.load_data()

    # Preprocess
    processor = DataPreprocessor()
    X, y = processor.preprocess(df)

    # Feature Engineering
    engineer = FeatureEngineering()

    X_train, X_test, y_train, y_test = engineer.split_data(X, y)

    X_train, X_test = engineer.scale_features(
        X_train,
        X_test
    )

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape :", X_test.shape)

    # Train Model
    model = IrisModel()

    print("\nTraining Model...")

    model.train(X_train, y_train)

    # Feature Importance
    model.feature_importance(X.columns)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    evaluator = ModelEvaluation()

    evaluator.evaluate(
        y_test,
        predictions
    )

    # Save Model
    model.save()

    print("\nModel Training Completed Successfully")


if __name__ == "__main__":
    main()