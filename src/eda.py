import os
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import DataLoader


class EDA:

    def __init__(self, dataframe):
        self.df = dataframe

    def dataset_info(self):

        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)

        print("\nShape :", self.df.shape)

        print("\nColumns :")

        print(self.df.columns.tolist())

        print("\nData Types :")

        print(self.df.dtypes)

        print("\nMissing Values :")

        print(self.df.isnull().sum())

        print("\nStatistics :")

        print(self.df.describe())

    def save_histograms(self):

        os.makedirs("outputs", exist_ok=True)

        self.df.hist(figsize=(10,8))

        plt.tight_layout()

        plt.savefig("outputs/histograms.png")

        plt.close()

        print("\n✅ Histogram Saved")


if __name__ == "__main__":

    loader = DataLoader("data/iris.csv")

    df = loader.load_data()

    analysis = EDA(df)

    analysis.dataset_info()

    analysis.save_histograms()