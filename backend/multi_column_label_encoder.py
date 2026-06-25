class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col] = {val: idx for idx, val in enumerate(sorted(X[col].unique()))}
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.encoders.items():
            X_copy[col] = X_copy[col].map(mapping)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
