#!/usr/bin/env bash
# Usage: publish_artifact.sh <local-zip-path> <s3-key>
set -e
if [[ -z "$S3_BUCKET" ]]; then
  echo "S3_BUCKET not set" >&2
  exit 1
fi
aws s3 cp "$1" "s3://$S3_BUCKET/$2"