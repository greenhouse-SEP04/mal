# Greenhouse ML Service

Stateless AWS Lambda service that trains or serves scikit-learn models on tabular data for a smart greenhouse.

## Repo structure

- `src/` : Lambda handler and ML logic
- `tests/` : unit tests for all components
- `scripts/` : helper scripts for build & publishing
- `.github/workflows/` : CI & CD pipelines
- `Dockerfile` : for local testing

## Quickstart

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
