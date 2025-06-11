# outputs.tf
output "lambda_function_arn" {
  description = "ARN of the retraining Lambda"
  value       = aws_lambda_function.retrain.arn
}

output "model_bucket_name" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_bucket.bucket
}

output "schedule_rule_arn" {
  description = "EventBridge rule ARN"
  value       = aws_cloudwatch_event_rule.daily.arn
}
