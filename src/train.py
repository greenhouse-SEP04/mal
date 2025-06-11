# train.py

import os
from src.data_ingestion import fetch_telemetry
from src.preprocessing import preprocess
from tuning import tune_models
from src.model import save_model, upload_to_s3
from sklearn.metrics import mean_squared_error, r2_score

def retrain_and_upload(api_url, device_mac, csv_path=None, enable_tuning=True):
    # 1) Load data
    df = fetch_telemetry(api_url, device_mac, limit=5000, csv_path=csv_path)

    # 2) Preprocess & split
    X_train, X_test, y_train, y_test = preprocess(df)

    # 3) Tune or default model
    if enable_tuning:
        model = tune_models(X_train, y_train)
    else:
        from src.model import build_model
        model = build_model()
        model.fit(X_train, y_train)

    # 4) Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"Test MSE: {mse:.3f}, R²: {r2:.3f}")

    # 5) Save locally
    local_path = '/tmp/model.joblib'
    save_model(model, local_path)

    # 6) Upload to S3
    bucket = os.environ['MODEL_BUCKET']
    s3_key = f"models/{device_mac}/model_{r2:.3f}.joblib"
    upload_to_s3(local_path, bucket, s3_key)

    return {'r2': r2, 'mse': mse, 's3_key': s3_key}

if __name__ == '__main__':
    # Example invocation for local testing
    API_URL    = "https://api.example.com"
    DEVICE_MAC = "AA:BB:CC:DD:EE:FF"
    CSV_PATH   = "../data/telemetry.csv"

    result = retrain_and_upload(API_URL, DEVICE_MAC,
                                csv_path=CSV_PATH, enable_tuning=True)
    print("Retraining job result:", result)
