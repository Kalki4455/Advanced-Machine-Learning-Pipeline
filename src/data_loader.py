import os
import pandas as pd


class DataLoader:
    """
    Responsible for loading datasets.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV dataset.
        """

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        df = pd.read_csv(self.file_path)

        return df


if __name__ == "__main__":

    loader = DataLoader("data/iris.csv")

    dataset = loader.load_data()

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print("\nFirst 5 Rows:\n")

    print(dataset.head())

    print("\nDataset Shape:", dataset.shape)

    print("\nColumns:")

    print(dataset.columns.tolist())

    print("\nData Types:\n")

    print(dataset.dtypes)