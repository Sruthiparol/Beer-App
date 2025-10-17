# utils.py

import pandas as pd
import numpy as np

# --- TARGET ENCODER CLASS ---
# This class must be defined in its own file so joblib can load it correctly 
# in both the training script and the main app.
class TargetEncoder:
    """Simple custom Target Encoder for 'country' column."""
    def __init__(self):
        self.mapping = {}
        self.global_mean = None
        self.col = None

    def fit(self, X: pd.Series, y: pd.Series):
        self.col = X.name
        df = pd.DataFrame({'X': X, 'y': y})
        self.global_mean = y.mean()
        self.mapping = df.groupby('X')['y'].mean().to_dict()
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        # Fill NaN (for new/unseen countries) with the global mean
        return X.map(self.mapping).fillna(self.global_mean)