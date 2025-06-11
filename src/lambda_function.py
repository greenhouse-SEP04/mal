# lambda_function.py

import os
from train import retrain_and_upload

def lambda_handler(event, context):
    """
    AWS Lambda entry point triggered by EventBridge to retrain model
    with fresh data and upload to S3.
    """
    api_url    = os.environ['API_URL']
    device_mac = os.environ['DEVICE_MAC']
    csv_path   = os.environ.get('CSV_PATH')  # optional

    result = retrain_and_upload(
        api_url,
        device_mac,
        csv_path=csv_path,
        enable_tuning=True
    )

    return {
        'statusCode': 200,
        'body': f"Retrain completed: R²={result['r2']:.3f}, key={result['s3_key']}"
    }
