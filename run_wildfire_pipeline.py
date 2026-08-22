from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire_risk_pipeline import predict_single_sample, train_and_compare_models


DEFAULT_SAMPLE = {
    "latitude": 21.1458,
    "longitude": 79.0882,
    "brightness": 332.6,
    "scan": 1.2,
    "track": 1.1,
    "confidence": 78,
    "bright_t31": 301.4,
    "month": 4,
    "year": 2024,
    "day_of_year": 109,
    "satellite": "Aqua",
    "instrument": "MODIS",
    "daynight": "D",
    "type": 0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a wildfire risk classifier.")
    parser.add_argument(
        "--data",
        default="df_2012_2024.csv",
        help="Path to the wildfire dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where evaluation reports and plots will be saved.",
    )
    parser.add_argument(
        "--sample-json",
        default=None,
        help="Optional JSON file containing a single custom sample for prediction.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=250000,
        help="Maximum rows to use for model training. Lower this if your PC runs out of memory.",
    )
    parser.add_argument(
        "--forest-weather-data",
        default=None,
        help="Optional CSV with dated forest weather points to merge by nearest location and date.",
    )
    args = parser.parse_args()

    results_df, best_artifact = train_and_compare_models(
        args.data,
        args.output_dir,
        max_rows=args.max_rows,
        forest_weather_path=args.forest_weather_data,
    )

    print("\nModel comparison")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_artifact.model_name}")
    print(f"Saved reports to: {Path(args.output_dir).resolve()}")
    print(f"Hotspot map: {(Path(args.output_dir) / 'wildfire_hotspot_map.html').resolve()}")

    sample = DEFAULT_SAMPLE
    if args.sample_json:
        with open(args.sample_json, "r", encoding="utf-8") as file:
            sample = json.load(file)

    prediction = predict_single_sample(best_artifact, sample)
    print("\nSample prediction")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
