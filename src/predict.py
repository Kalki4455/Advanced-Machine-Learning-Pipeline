import os
import joblib
import pandas as pd


class Predictor:

    def __init__(self):
        model_path = os.path.join("models", "iris_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Train the model first."
            )

        self.model = joblib.load(model_path)

    def predict(self, sepal_length, sepal_width,
                petal_length, petal_width):

        sample = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width"
            ]
        )

        prediction = self.model.predict(sample)

        return prediction[0]


if __name__ == "__main__":

    predictor = Predictor()

    result = predictor.predict(
        5.1,
        3.5,
        1.4,
        0.2
    )

    print("=" * 50)
    print("Prediction")
    print("=" * 50)
    print(result)