from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_categorical_dtype, is_object_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RISK_LABELS = ["Low", "Moderate", "High", "Extreme"]
RISK_COLORS = {
    "Low": "#2E8B57",
    "Moderate": "#E6B800",
    "High": "#F57C00",
    "Extreme": "#C62828",
}


@dataclass
class TrainingArtifacts:
    model_name: str
    pipeline: Pipeline
    metrics: dict[str, Any]
    predictions: pd.Series
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]


def assign_risk_level(frp: float) -> str:
    if frp < 10:
        return "Low"
    if frp < 25:
        return "Moderate"
    if frp < 100:
        return "High"
    return "Extreme"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    dtype_map = {
        "latitude": "float32",
        "longitude": "float32",
        "brightness": "float32",
        "scan": "float32",
        "track": "float32",
        "confidence": "float32",
        "bright_t31": "float32",
        "frp": "float32",
        "satellite": "category",
        "instrument": "category",
        "daynight": "category",
        "type": "int8",
    }
    df = pd.read_csv(csv_path, dtype=dtype_map, low_memory=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date", "frp"]).copy()
    df["month"] = df["acq_date"].dt.month
    df["year"] = df["acq_date"].dt.year
    df["day_of_year"] = df["acq_date"].dt.dayofyear
    df["risk_level"] = df["frp"].apply(assign_risk_level)
    return df


def load_forest_weather_dataset(csv_path: str | Path) -> pd.DataFrame:
    dtype_map = {
        "WindSpeed": "float32",
        "Humidity": "float32",
        "Temperature": "float32",
        "PRECTOTCORR": "float32",
        "Zone": "category",
        "Latitude": "float32",
        "Longitude": "float32",
    }
    df = pd.read_csv(csv_path, dtype=dtype_map, low_memory=True)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE", "Latitude", "Longitude"]).copy()
    df["point_id"] = (
        df["Latitude"].round(4).astype(str) + "_" + df["Longitude"].round(4).astype(str)
    )
    return df


def merge_nearest_forest_weather(
    wildfire_df: pd.DataFrame,
    forest_weather_df: pd.DataFrame,
) -> pd.DataFrame:
    wildfire_df = wildfire_df.copy()

    overlap_start = forest_weather_df["DATE"].min()
    overlap_end = forest_weather_df["DATE"].max()
    wildfire_df = wildfire_df[
        (wildfire_df["acq_date"] >= overlap_start) & (wildfire_df["acq_date"] <= overlap_end)
    ].copy()

    forest_points = (
        forest_weather_df[["point_id", "Latitude", "Longitude", "Zone"]]
        .drop_duplicates(subset=["point_id"])
        .reset_index(drop=True)
    )

    point_coords = np.deg2rad(forest_points[["Latitude", "Longitude"]].to_numpy())
    fire_coords = np.deg2rad(wildfire_df[["latitude", "longitude"]].to_numpy())

    tree = BallTree(point_coords, metric="haversine")
    distance_radians, index_array = tree.query(fire_coords, k=1)
    matched_points = forest_points.iloc[index_array.flatten()].reset_index(drop=True)

    wildfire_df = wildfire_df.reset_index(drop=True)
    wildfire_df["point_id"] = matched_points["point_id"]
    wildfire_df["forest_zone"] = matched_points["Zone"].astype("category")
    wildfire_df["forest_point_distance_km"] = (distance_radians.flatten() * 6371.0).astype("float32")

    forest_daily = forest_weather_df.rename(
        columns={
            "DATE": "acq_date",
            "WindSpeed": "forest_wind_speed",
            "Humidity": "forest_humidity",
            "Temperature": "forest_temperature",
            "PRECTOTCORR": "forest_precipitation",
        }
    )[
        [
            "acq_date",
            "point_id",
            "forest_wind_speed",
            "forest_humidity",
            "forest_temperature",
            "forest_precipitation",
        ]
    ]

    wildfire_df = wildfire_df.merge(forest_daily, on=["acq_date", "point_id"], how="left")
    return wildfire_df


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "confidence",
        "bright_t31",
        "month",
        "year",
        "day_of_year",
        "satellite",
        "instrument",
        "daynight",
        "type",
    ]
    optional_columns = [
        "forest_wind_speed",
        "forest_humidity",
        "forest_temperature",
        "forest_precipitation",
        "forest_zone",
        "forest_point_distance_km",
    ]
    feature_columns.extend([col for col in optional_columns if col in df.columns])
    model_df = df[feature_columns + ["risk_level"]].copy()
    x = model_df[feature_columns]
    y = model_df["risk_level"]
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [
        col for col in x.columns if is_object_dtype(x[col]) or is_categorical_dtype(x[col])
    ]
    numeric_features = [col for col in x.columns if col not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=4, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=1,
            max_samples=0.5,
        ),
    }


def reduce_dataset_for_training(
    x: pd.DataFrame,
    y: pd.Series,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(x) <= max_rows:
        return x, y

    sampled_indices = (
        y.groupby(y, group_keys=False)
        .apply(lambda series: series.sample(frac=max_rows / len(y), random_state=random_state))
        .index
    )
    return x.loc[sampled_indices].copy(), y.loc[sampled_indices].copy()


def train_and_compare_models(
    csv_path: str | Path,
    output_dir: str | Path = "outputs",
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int | None = 250_000,
    forest_weather_path: str | Path | None = None,
) -> tuple[pd.DataFrame, TrainingArtifacts]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    if forest_weather_path is not None:
        forest_weather_df = load_forest_weather_dataset(forest_weather_path)
        df = merge_nearest_forest_weather(df, forest_weather_df)
    risk_zone_summary = create_risk_zone_summary(df, output_path / "risk_zone_summary.csv")
    create_hotspot_map(risk_zone_summary, output_path / "wildfire_hotspot_map.html")
    x, y = build_feature_frame(df)
    x, y = reduce_dataset_for_training(x, y, max_rows=max_rows, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(x)
    model_summaries: list[dict[str, Any]] = []
    best_artifact: TrainingArtifacts | None = None

    for model_name, estimator in build_models(random_state=random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pd.Series(pipeline.predict(x_test), index=y_test.index, name="predicted_risk_level")

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=RISK_LABELS)
        accuracy = accuracy_score(y_test, predictions)

        summary = {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "macro_precision": round(report["macro avg"]["precision"], 4),
            "macro_recall": round(report["macro avg"]["recall"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
        }
        model_summaries.append(summary)

        artifact = TrainingArtifacts(
            model_name=model_name,
            pipeline=pipeline,
            metrics=summary,
            predictions=predictions,
            confusion_matrix=matrix.tolist(),
            classification_report=report,
        )

        if best_artifact is None or artifact.metrics["macro_f1"] > best_artifact.metrics["macro_f1"]:
            best_artifact = artifact

    assert best_artifact is not None

    results_df = pd.DataFrame(model_summaries).sort_values(
        by=["macro_f1", "accuracy"], ascending=False
    )
    results_df.to_csv(output_path / "model_comparison.csv", index=False)

    prediction_output = x_test.copy()
    prediction_output["actual_risk_level"] = y_test
    prediction_output["predicted_risk_level"] = best_artifact.predictions
    prediction_output.to_csv(output_path / "best_model_predictions.csv", index=False)

    with (output_path / "best_model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_artifact.model_name,
                "summary_metrics": best_artifact.metrics,
                "classification_report": best_artifact.classification_report,
                "confusion_matrix": {
                    "labels": RISK_LABELS,
                    "values": best_artifact.confusion_matrix,
                },
            },
            file,
            indent=2,
        )

    plot_model_scores(results_df, output_path / "model_comparison.png")
    plot_confusion(best_artifact, output_path / "best_model_confusion_matrix.png")

    return results_df, best_artifact


def create_risk_zone_summary(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    zone_df = df.copy()
    zone_df["lat_zone"] = zone_df["latitude"].round(1)
    zone_df["lon_zone"] = zone_df["longitude"].round(1)

    summary = (
        zone_df.groupby(["lat_zone", "lon_zone"], as_index=False)
        .agg(
            wildfire_events=("frp", "size"),
            avg_frp=("frp", "mean"),
            max_frp=("frp", "max"),
            avg_brightness=("brightness", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .sort_values(by=["wildfire_events", "avg_frp"], ascending=False)
    )
    summary["dominant_risk_level"] = summary["avg_frp"].apply(assign_risk_level)
    summary.to_csv(output_path, index=False)
    return summary


def create_hotspot_map(summary_df: pd.DataFrame, output_path: str | Path, top_n: int = 1500) -> None:
    hotspot_df = summary_df.head(top_n).copy()
    map_center = [hotspot_df["lat_zone"].mean(), hotspot_df["lon_zone"].mean()]
    hotspot_map = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")

    for _, row in hotspot_df.iterrows():
        risk_level = row["dominant_risk_level"]
        event_scale = min(24, max(5, row["wildfire_events"] ** 0.5 / 2.5))
        popup_html = f"""
        <div style="width: 240px;">
            <h4 style="margin-bottom: 8px;">Wildfire Hotspot</h4>
            <b>Risk Level:</b> {risk_level}<br>
            <b>Latitude Zone:</b> {row['lat_zone']}<br>
            <b>Longitude Zone:</b> {row['lon_zone']}<br>
            <b>Wildfire Events:</b> {int(row['wildfire_events'])}<br>
            <b>Average FRP:</b> {row['avg_frp']:.2f}<br>
            <b>Maximum FRP:</b> {row['max_frp']:.2f}<br>
            <b>Average Brightness:</b> {row['avg_brightness']:.2f}<br>
            <b>Average Confidence:</b> {row['avg_confidence']:.2f}
        </div>
        """
        folium.CircleMarker(
            location=[row["lat_zone"], row["lon_zone"]],
            radius=event_scale,
            color=RISK_COLORS[risk_level],
            fill=True,
            fill_color=RISK_COLORS[risk_level],
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=(
                f"{risk_level} risk | Events: {int(row['wildfire_events'])} | "
                f"Avg FRP: {row['avg_frp']:.1f}"
            ),
        ).add_to(hotspot_map)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 180px;
        background: white;
        border: 1px solid #bbb;
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
        z-index: 9999;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <b>Risk Levels</b><br>
        <span style="color:#2E8B57;">●</span> Low<br>
        <span style="color:#E6B800;">●</span> Moderate<br>
        <span style="color:#F57C00;">●</span> High<br>
        <span style="color:#C62828;">●</span> Extreme<br><br>
        Circle size reflects wildfire event count.
    </div>
    """
    hotspot_map.get_root().html.add_child(folium.Element(legend_html))
    hotspot_map.save(str(output_path))


def plot_model_scores(results_df: pd.DataFrame, output_path: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = results_df.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Wildfire Risk Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion(artifact: TrainingArtifacts, output_path: str | Path) -> None:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        pd.DataFrame(artifact.confusion_matrix, index=RISK_LABELS, columns=RISK_LABELS),
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {artifact.model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def predict_single_sample(
    artifact: TrainingArtifacts,
    sample: dict[str, Any],
) -> dict[str, Any]:
    sample_df = pd.DataFrame([sample])
    expected_columns = list(artifact.pipeline.named_steps["preprocessor"].feature_names_in_)
    for column in expected_columns:
        if column not in sample_df.columns:
            sample_df[column] = None
    sample_df = sample_df[expected_columns]
    prediction = artifact.pipeline.predict(sample_df)[0]
    probabilities = artifact.pipeline.predict_proba(sample_df)[0]
    return {
        "predicted_risk_level": prediction,
        "class_probabilities": {
            label: round(float(probability), 4)
            for label, probability in zip(artifact.pipeline.named_steps["model"].classes_, probabilities)
        },
    }
