from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:

    def __init__(self):
        self.scaler = StandardScaler()

    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled