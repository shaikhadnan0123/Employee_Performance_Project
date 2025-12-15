import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        X = X.copy()
        cols = self.columns if self.columns is not None else X.columns
        for col in cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = le.transform(X[col])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
