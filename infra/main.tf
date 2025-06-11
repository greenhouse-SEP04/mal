# main.tf
# 1) Create S3 bucket
resource "aws_s3_bucket" "model_bucket" {
  bucket = var.bucket_name
  acl    = "private"

  tags = {
    Name = "ModelBucket"
  }
}

# 2) Package Lambda code from local folder
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = var.source_dir
  output_path = "${path.module}/lambda_package.zip"
}

# 3) IAM role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
      Effect    = "Allow"
    }]
  })
}

# 4) Attach AWSLambdaBasicExecutionRole (CloudWatch logs)
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# 5) Inline policy for S3 PutObject
resource "aws_iam_role_policy" "lambda_s3_put" {
  name = "${var.function_name}-s3-put"

  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject"]
      Resource = ["${aws_s3_bucket.model_bucket.arn}/*"]
    }]
  })
}

# 6) Lambda function
resource "aws_lambda_function" "retrain" {
  function_name = var.function_name
  filename      = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  handler       = var.handler
  runtime       = var.lambda_runtime
  role          = aws_iam_role.lambda_role.arn
  memory_size   = 512
  timeout       = 900

  environment {
    variables = {
      API_URL      = var.api_url
      DEVICE_MAC   = var.device_mac
      MODEL_BUCKET = aws_s3_bucket.model_bucket.bucket
    }
  }
}

# 7) CloudWatch EventBridge rule for daily schedule
resource "aws_cloudwatch_event_rule" "daily" {
  name                = "${var.function_name}-schedule"
  schedule_expression = var.event_schedule
  description         = "Daily retraining trigger"
}

# 8) Hook the rule to Lambda
resource "aws_cloudwatch_event_target" "to_lambda" {
  rule      = aws_cloudwatch_event_rule.daily.name
  target_id = "LambdaTarget"
  arn       = aws_lambda_function.retrain.arn
}

# 9) Grant EventBridge permission to invoke the function
resource "aws_lambda_permission" "allow_event" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.retrain.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily.arn
}
