"""
Aggregate telemetry from multiple devices into one raw CSV.
"""
import argparse
import pandas as pd
from .data_fetcher import ApiClient
from . import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="aggregated_telemetry.csv")
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]
    orig = config.DEVICE_MAC
    frames = []
    client = ApiClient()

    for dev in devices:
        config.DEVICE_MAC = dev
        try:
            df = client.latest_telemetry(limit=args.limit)
            df["DeviceMac"] = dev
            frames.append(df)
            print(f"Fetched {len(df)} rows for {dev}")
        except Exception as e:
            print(f"Warning: {dev} failed: {e}")
    config.DEVICE_MAC = orig

    if not frames:
        print("No data fetched.")
        return
    agg = pd.concat(frames, ignore_index=True)
    agg.sort_values(["Timestamp","DeviceMac"], inplace=True)
    agg.to_csv(args.output, index=False)
    print(f"Wrote {len(agg)} rows to {args.output}")

if __name__ == "__main__":
    main()