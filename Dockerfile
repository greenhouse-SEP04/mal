# mal/Dockerfile.lambda
FROM public.ecr.aws/lambda/python:3.11

# Optional: faster scientific stack builds: pre-install wheels
# RUN pip install --no-cache-dir numpy pandas scikit-learn joblib boto3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and set handler
COPY src ./src
# Tell the runtime which handler to call
CMD [ "src/greenhouse_ml_service.handler" ]
