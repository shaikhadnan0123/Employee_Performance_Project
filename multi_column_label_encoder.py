class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # your transformation logic here
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
