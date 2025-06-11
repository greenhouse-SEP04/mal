# model.py

import joblib
import boto3

def build_model():
    """
    Default pipeline: standard scaler + RandomForestRegressor.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])

def save_model(pipe, path: str = 'model.joblib'):
    joblib.dump(pipe, path)

def load_model(path: str = 'model.joblib'):
    return joblib.load(path)

def upload_to_s3(local_path: str, bucket: str, s3_key: str):
    """
    Upload a file to S3.
    """
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_key)
