import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


MODEL_PATH = "models/posture_classifier_eval.pkl"

ROBOFLOW_CSV = "data/processed/roboflow_posture_features.csv"
REAL_CSV = "data/processed/real_video_features.csv"
SYNTHETIC_CSV = "data/processed/synthetic_good_bad_features.csv"
BORDERLINE_CSV = "data/processed/synthetic_borderline_features.csv"


def load_csv(path, source_name):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["data_source"] = source_name

    return df


def create_group_id(row):
    source_file = str(
        row.get("source_file", "unknown")
    )

    if row["data_source"] in {
        "real_video",
        "synthetic_video"
    }:
        return (
            f"{row['data_source']}::"
            f"{source_file}"
        )

    source_dataset = str(
        row.get(
            "source_dataset",
            "unknown_dataset"
        )
    )

    return (
        f"roboflow::"
        f"{source_dataset}::"
        f"{source_file}"
    )


def prepare_features(df, feature_names):
    X = df.copy()

    for feature in feature_names:
        if feature not in X.columns:
            X[feature] = np.nan

    X = X[feature_names]

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    return X


def evaluate_subset(
    name,
    df,
    model,
    feature_names
):
    if df.empty:
        return

    df = df[
        df["label"].isin(
            ["good", "bad"]
        )
    ].copy()

    if df.empty:
        return

    X = prepare_features(
        df,
        feature_names
    )

    y = df["label"]

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print(f"Samples: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0
        )
    )


def evaluate_each_video(
    real_df,
    model,
    feature_names
):
    print("\n" + "=" * 55)
    print("UNSEEN REAL VIDEOS")
    print("=" * 55)

    results = []

    for source_file, group in real_df.groupby(
        "source_file"
    ):
        if group.empty:
            continue

        X = prepare_features(
            group,
            feature_names
        )

        y = group["label"]

        predictions = model.predict(X)

        accuracy = accuracy_score(
            y,
            predictions
        )

        results.append({
            "video": source_file,
            "label": y.iloc[0],
            "samples": len(group),
            "accuracy": accuracy,
        })

    if results:
        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            "accuracy"
        )

        print(
            results_df.to_string(
                index=False
            )
        )


def evaluate_face_availability(
    holdout_df,
    model,
    feature_names
):
    print("\n" + "=" * 55)
    print("UNSEEN: POSE + FACE VS POSE ONLY")
    print("=" * 55)

    for face_value, name in [
        (1, "Pose + Face"),
        (0, "Pose only"),
    ]:
        subset = holdout_df[
            holdout_df[
                "face_available"
            ] == face_value
        ]

        evaluate_subset(
            name,
            subset,
            model,
            feature_names
        )


def evaluate_borderline(
    borderline_df,
    model,
    feature_names
):
    if borderline_df.empty:
        return

    print("\n" + "=" * 55)
    print("BORDERLINE PROBABILITY ANALYSIS")
    print("=" * 55)

    X = prepare_features(
        borderline_df,
        feature_names
    )

    probabilities = model.predict_proba(X)

    classes = list(
        model.classes_
    )

    good_index = classes.index(
        "good"
    )

    good_probability = (
        probabilities[:, good_index]
    )

    borderline_df = borderline_df.copy()

    borderline_df[
        "good_probability"
    ] = good_probability

    summary = (
        borderline_df[
            [
                "source_file",
                "good_probability"
            ]
        ]
        .groupby(
            "source_file"
        )
        .agg(
            [
                "mean",
                "min",
                "max"
            ]
        )
        .round(3)
    )

    print(summary)


def main():

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]
    feature_names = package["features"]
    holdout_groups = set(
        package["holdout_groups"]
    )

    roboflow_df = load_csv(
        ROBOFLOW_CSV,
        "roboflow"
    )

    real_df = load_csv(
        REAL_CSV,
        "real_video"
    )

    synthetic_df = load_csv(
        SYNTHETIC_CSV,
        "synthetic_video"
    )

    borderline_df = load_csv(
        BORDERLINE_CSV,
        "borderline"
    )

    all_df = pd.concat(
        [
            roboflow_df,
            real_df,
            synthetic_df
        ],
        ignore_index=True,
        sort=False
    )

    all_df["group_id"] = all_df.apply(
        create_group_id,
        axis=1
    )

    holdout_df = all_df[
        all_df[
            "group_id"
        ].isin(
            holdout_groups
        )
    ].copy()

    print("\n" + "=" * 60)
    print("HOLDOUT-ONLY DIAGNOSTIC DATA")
    print("=" * 60)

    print(
        f"Holdout rows: "
        f"{len(holdout_df)}"
    )

    print(
        f"Holdout groups: "
        f"{holdout_df['group_id'].nunique()}"
    )

    evaluate_subset(
        "UNSEEN ROBOFLOW",
        holdout_df[
            holdout_df[
                "data_source"
            ] == "roboflow"
        ],
        model,
        feature_names
    )

    evaluate_subset(
        "UNSEEN REAL VIDEO",
        holdout_df[
            holdout_df[
                "data_source"
            ] == "real_video"
        ],
        model,
        feature_names
    )

    evaluate_subset(
        "UNSEEN SYNTHETIC",
        holdout_df[
            holdout_df[
                "data_source"
            ] == "synthetic_video"
        ],
        model,
        feature_names
    )

    evaluate_face_availability(
        holdout_df,
        model,
        feature_names
    )

    unseen_real = holdout_df[
        holdout_df[
            "data_source"
        ] == "real_video"
    ]

    evaluate_each_video(
        unseen_real,
        model,
        feature_names
    )

    # Borderline stays separate from holdout.
    evaluate_borderline(
        borderline_df,
        model,
        feature_names
    )


if __name__ == "__main__":
    main()