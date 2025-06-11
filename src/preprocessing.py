# preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame) -> tuple:
    """
    Clean and feature-engineer telemetry DataFrame. Split into train/test.
    """
    # Drop rows missing core features
    df = df.dropna(subset=['soil', 'lux', 'temperature', 'humidity'])

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['soil_roll3'] = df['soil'].rolling(window=3).mean().fillna(method='bfill')

    # Define X and y
    X = df[['lux', 'temperature', 'humidity', 'hour', 'soil_roll3']]
    y = df['soil']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
