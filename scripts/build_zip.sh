#!/bin/bash

set -e

mkdir -p build
cd src

# Rename and zip
cp greenhouse_ml_service.py ../build/handler.py
cd ../build

# Install requirements locally into zip (optional for Lambda)
pip install --target . -r ../requirements.txt

# Package everything into ZIP
zip -r ml_service.zip .
