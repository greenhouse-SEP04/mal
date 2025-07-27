#!/usr/bin/env bash
# Usage: publish_artifact.sh <local-zip-path> <s3-key> [lambda-function-name]

set -euo pipefail

ZIP_PATH="$1"
S3_KEY="$2"
FUNCTION_NAME="${3:-}"

if [[ -z "${S3_BUCKET:-}" ]]; then
  echo "❌ S3_BUCKET environment variable not set" >&2
  exit 1
fi

echo "☁️ Uploading artifact to S3..."
aws s3 cp "$ZIP_PATH" "s3://$S3_BUCKET/$S3_KEY"

# Optional: update Lambda function if name is provided
if [[ -n "$FUNCTION_NAME" ]]; then
  echo "🔁 Updating Lambda function: $FUNCTION_NAME"
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --s3-bucket "$S3_BUCKET" \
    --s3-key "$S3_KEY"
  echo "✅ Lambda function updated"
else
  echo "ℹ️ Lambda function name not provided, skipping update"
fi
