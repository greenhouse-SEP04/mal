# variables.tf
variable "aws_region" {
  description = "AWS region to deploy into"
  default     = "eu-central-1"
}

variable "function_name" {
  description = "Name of the Lambda function"
  default     = "ml_retrain_function"
}

variable "lambda_runtime" {
  description = "Lambda runtime"
  default     = "python3.9"
}

variable "handler" {
  description = "Lambda handler (file.function)"
  default     = "lambda_function.lambda_handler"
}

variable "source_dir" {
  description = "Path to the local ml_service folder"
  default     = "../ml_service"
}

variable "bucket_name" {
  description = "S3 bucket to store model artifacts"
  default     = "my-ml-model-bucket"
}

variable "event_schedule" {
  description = "CloudWatch Events schedule expression"
  default     = "rate(1 day)"
}

variable "api_url" {
  description = "API endpoint for retraining"
  default     = "https://api.example.com"
}

variable "device_mac" {
  description = "Device MAC address"
  default     = "AA:BB:CC:DD:EE:FF"
}
