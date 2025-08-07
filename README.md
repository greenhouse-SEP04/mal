# 🌱 Greenhouse Machine‑Learning Service (`mal`)

This repository contains the **machine‑learning micro‑service** for the SEP4 Greenhouse project.  It trains and serves lightweight ML models that decide
when to **water the plants** and when to **open the ventilation** based on telemetry streamed from the IoT node (`ews`).  The service is designed for low‑latency inference on AWS Lambda while retaining reproducible, version‑controlled builds via GitHub Actions and Terraform.

> **Key facts**
>
> |          | Details                                                        |
> | -------- | -------------------------------------------------------------- |
> | Language | Python 3.11                                                    |
> | Runtime  | AWS Lambda (zip + layers)                                      |
> | ML libs  | *scikit‑learn*, *pandas*, *numpy*                              |
> | CI/CD    | GitHub Actions → Release assets → Terraform deploy             |
> | Infra    | S3 (telemetry+artifacts), API Gateway, EventBridge, CloudWatch |

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Quick Start](#quick-start)
4. [Data Schema](#data-schema)
5. [Training & Inference Flows](#training--inference-flows)
6. [Configuration](#configuration)
7. [DevOps Pipeline](#devops-pipeline)
8. [Testing](#testing)
9. [Contributing](#contributing)

---

## Architecture

```mermaid
flowchart TD
    A[IoT node (AVR\n`ews`)] -- CSV telemetry --> S3[(Telemetry bucket)]
    subgraph ML Service
        B(API Gateway \n POST /v1/predict)
        C[Lambda handler (src/handler.py)]
        D[(ML models\n.joblib in S3)]
    end
    B --> C
    C -- get/put --> D
    E[GitHub Actions CD] --> F((Release assets)) --> Terraform --> C
    EventBridge -->|hourly| C
```

* **Inference**: The AVR device POSTs a JSON event to **/v1/predict**; API Gateway invokes the Lambda which loads the latest model from S3 and returns a binary decision (on/off) plus probability.
* **Training**: An hourly EventBridge trigger (or manual API call) makes the Lambda download the most recent CSV telemetry, fit a Random Forest model for each target *(water, vent)*, and write the model + metrics back to S3.

---

## Repository Layout

```
mal/
├── .github/workflows/       # CI/CD pipeline (build layers + release)
│   └── cd.yml
├── scripts/                 # build helpers
│   ├── build_layers.sh      # 🐍 numpy/pandas/sklearn → layer zips
│   ├── build_zip.sh         # handler → ml_service.zip
│   └── publish_artifact.sh  # optional S3 + Lambda update
├── src/
│   └── handler.py           # Lambda entry‑point (train / predict / ping)
├── requirements.txt         # runtime deps (mirrors build_layers versions)
└── README.md                # you are here 🙌
```

---

## Quick Start

### 1. Local dev (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest   # run unit tests (WIP)
```

### 2. Build artefacts manually

```bash
./scripts/build_layers.sh   # creates .lambda_layers/{sk1,sk2}.zip
./scripts/build_zip.sh      # creates build/ml_service.zip
```

### 3. Invoke the handler locally

Use **docker‑lambda** or **AWS SAM CLI**:

```bash
sam local invoke -e events/predict_water.json \
  --env-vars sam.env.json
```

### 4. Deploy (CI‑driven)

1. Push to `main` → **cd.yml** builds layers + zip and publishes a GitHub Release.
2. The **terraform** repo downloads those release assets and provisions layers, Lambda, EventBridge rule, and the public HTTP endpoint.

> ℹ️ All AWS resources live in the `greenhouse-ml` Terraform module for ease of reuse.

---

## Data Schema

| Column        | Type     | Description                     |
| ------------- | -------- | ------------------------------- |
| `DeviceMac`   | string   | MAC address of IoT node         |
| `Timestamp`   | ISO 8601 | Sample time (UTC)               |
| `Temperature` | °C       | Air temperature                 |
| `Humidity`    | % RH     | Air humidity                    |
| `Soil`        | %        | Soil moisture (0 dry – 100 wet) |
| `Lux`         | lux      | Light intensity                 |
| `Level`       | cm       | Tank water level                |
| `Motion`      | 0/1      | PIR motion flag                 |
| `Tamper`      | 0/1      | ADXL345 tamper flag             |
| `AccelX/Y/Z`  | int      | Raw accelerometer axes          |
| `MlWater`     | 0/1      | Ground‑truth label (optional)   |
| `MlVent`      | 0/1      | Ground‑truth label (optional)   |

If labels are missing the Lambda derives **heuristic labels** (water ↔ `Soil <= 40`, vent ↔ `Humidity <= 45`) so it can train even with unlabeled data.

---

## Training & Inference Flows

### Train

```jsonc
{
  "action": "train",
  "greenhouse_id": "gh-001",
  "s3_uri": "s3://<bucket>/ml/training.csv", // OR bucket+key
  "target": "water" // water|vent|omit→both
}
```

* **Model**: `RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample")`
* **Outputs**: `ml/models/<gh>/<target>_rf_cls.joblib` + metrics JSON under `ml/metrics/…`.

### Predict

```jsonc
{
  "action": "predict",
  "greenhouse_id": "gh-001",
  "target": "vent",
  "features": {
    "Temperature": 25,
    "Humidity": 50,
    "Soil": 55,
    "Lux": 1200,
    "Level": 18,
    "Motion": 0,
    "Tamper": 0,
    "AccelX": 8,
    "AccelY": -3,
    "AccelZ": 1023,
    "Hour": 14
  }
}
```

Response:

```jsonc
{
  "ok": true,
  "result": {
    "target": "vent",
    "prediction": 0,
    "probability": 0.27
  }
}
```

---

## Configuration

| ENV var            | Default   | Purpose                        |
| ------------------ | --------- | ------------------------------ |
| `S3_BUCKET`        | *(none)*  | Telemetry & model bucket       |
| `MIN_SAMPLES`      | `10`      | Minimum rows required to train |
| `AWS_ENDPOINT_URL` | *(unset)* | Override for **LocalStack**    |

---

## DevOps Pipeline

### GitHub Actions – `.github/workflows/cd.yml`

1. **Checkout**
2. **Set up Python 3.11**
3. **Cache pip** deps
4. **Build Lambda layers** (`build_layers.sh`)
5. **Build handler ZIP** (`build_zip.sh`)
6. **Create GitHub Release** with assets
7. *(Optional)* Publish artifacts directly to S3 + update Lambda

### Terraform (sibling repo)

* Downloads release assets → `local_file` resources
* Creates **Lambda layers** & **function**
* Exposes **/v1/predict** via API Gateway (HTTP API)
* Schedules hourly retraining via EventBridge

---


For integration you can spin up **LocalStack**:

```bash
localstack start -d
export AWS_ENDPOINT_URL=http://localhost:4566
python src/handler.py  # invoke however you like
```

---

## Contributing

1. Fork & create a feature branch.
2. Follow the [commit convention](https://www.conventionalcommits.org/) – this feeds the release notes.
3. Run `pre‑commit run ‑‑all-files` before pushing.
4. Open a PR – CI must be ✔️ for merge.

---

## License

Distributed under the MIT License.  See `LICENSE` for details.

---

> *SEP4 — 2025 • VIA University College*
