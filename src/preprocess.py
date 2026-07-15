import pandas as pd


class DataPreprocessor:

    def preprocess(self, df: pd.DataFrame):

        print("=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)

        df = df.dropna()
        df = df.drop_duplicates()

        # Drop Id column
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        # Features
        X = df.drop(columns=["Species"])

        # Target
        y = df["Species"]

        print("\nDataset Shape :", df.shape)
        print("Features Shape :", X.shape)
        print("Target Shape :", y.shape)

        print("\nFeature Columns:")
        print(X.columns.tolist())

        return X, y