# 🌱 Smart Greenhouse – ML Service

A minimal Python project that trains per-plant random-forest models to forecast soil moisture and runs as an AWS Lambda on a cron schedule.

## Quick start
```bash
pip install -r requirements.txt
export API_BASE=https://api.example.com/v1
export API_USER=admin
export API_PASS=secret
export DEVICE_MAC=AA:BB:CC:DD:EE:FF
# Train model
python -m ml_service.train
# Run aggregator (global data)
python -m ml_service.aggregate --devices AA:BB:CC:DD:EE:FF,11:22:33:44:55 --limit 10000 --output aggregated_telemetry.csv
