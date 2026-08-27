import os
import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = "models/posture_classifier_eval.pkl"

ROBOFLOW_CSV = "data/processed/roboflow_posture_features.csv"
REAL_CSV = "data/processed/real_video_features.csv"
SYNTHETIC_CSV = "data/processed/synthetic_good_bad_features.csv"
BORDERLINE_CSV = "data/processed/synthetic_borderline_features.csv"

CALIBRATOR_PATH = "models/posture_score_calibrator.pkl"


# =========================================================
# HELPERS
# =========================================================

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


def get_good_probabilities(
    df,
    model,
    feature_names
):

    X = prepare_features(
        df,
        feature_names
    )

    probabilities = model.predict_proba(X)

    classes = list(
        model.classes_
    )

    good_index = classes.index(
        "good"
    )

    return probabilities[
        :,
        good_index
    ]


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------
    # Load evaluation model
    # -----------------------------------------

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    feature_names = package[
        "features"
    ]

    holdout_groups = set(
        package["holdout_groups"]
    )


    # -----------------------------------------
    # Load good/bad data
    # -----------------------------------------

    roboflow = load_csv(
        ROBOFLOW_CSV,
        "roboflow"
    )

    real = load_csv(
        REAL_CSV,
        "real_video"
    )

    synthetic = load_csv(
        SYNTHETIC_CSV,
        "synthetic_video"
    )


    all_binary = pd.concat(
        [
            roboflow,
            real,
            synthetic
        ],
        ignore_index=True,
        sort=False
    )


    all_binary["group_id"] = (
        all_binary.apply(
            create_group_id,
            axis=1
        )
    )


    # Only use unseen holdout groups
    calibration_binary = all_binary[
        all_binary["group_id"].isin(
            holdout_groups
        )
    ].copy()


    calibration_binary[
        "good_probability"
    ] = get_good_probabilities(
        calibration_binary,
        model,
        feature_names
    )


    # -----------------------------------------
    # Reduce to GROUP level
    # -----------------------------------------

    binary_groups = (
        calibration_binary
        .groupby(
            [
                "group_id",
                "label"
            ]
        )["good_probability"]
        .mean()
        .reset_index()
    )


    binary_groups[
        "target_quality"
    ] = binary_groups[
        "label"
    ].map({
        "bad": 0.0,
        "good": 1.0,
    })


    # =========================================
    # BORDERLINE DATA
    # =========================================

    borderline = load_csv(
        BORDERLINE_CSV,
        "borderline"
    )


    borderline[
        "good_probability"
    ] = get_good_probabilities(
        borderline,
        model,
        feature_names
    )


    borderline_groups = (
        borderline
        .groupby(
            "source_file"
        )["good_probability"]
        .mean()
        .reset_index()
    )


    # Borderline represents the middle of
    # posture quality for calibration.
    borderline_groups[
        "target_quality"
    ] = 0.5


    # =========================================
    # COMBINE CALIBRATION POINTS
    # =========================================

    binary_points = pd.DataFrame({
        "probability":
            binary_groups[
                "good_probability"
            ],

        "target":
            binary_groups[
                "target_quality"
            ]
    })


    borderline_points = pd.DataFrame({
        "probability":
            borderline_groups[
                "good_probability"
            ],

        "target":
            borderline_groups[
                "target_quality"
            ]
    })


    calibration_points = pd.concat(
        [
            binary_points,
            borderline_points
        ],
        ignore_index=True
    )


    # =========================================
    # FIT MONOTONIC CALIBRATOR
    # =========================================

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )


    calibrator.fit(
        calibration_points[
            "probability"
        ],
        calibration_points[
            "target"
        ]
    )


    # =========================================
    # SAVE
    # =========================================

    joblib.dump(
        calibrator,
        CALIBRATOR_PATH
    )


    print("\n" + "=" * 60)
    print("SCORE CALIBRATION COMPLETE")
    print("=" * 60)


    print(
        f"Binary holdout groups used: "
        f"{len(binary_groups)}"
    )

    print(
        f"Borderline videos used: "
        f"{len(borderline_groups)}"
    )


    # =========================================
    # BORDERLINE BEFORE / AFTER
    # =========================================

    print("\nBorderline calibration results:")


    for _, row in borderline_groups.iterrows():

        raw_probability = (
            row[
                "good_probability"
            ]
        )

        calibrated_quality = float(
            calibrator.predict(
                [raw_probability]
            )[0]
        )


        print(
            f"\n{row['source_file']}"
        )

        print(
            f"  Raw good probability: "
            f"{raw_probability:.3f}"
        )

        print(
            f"  Calibrated score: "
            f"{calibrated_quality * 100:.1f}"
        )


    # =========================================
    # GOOD / BAD GROUP SUMMARY
    # =========================================

    binary_groups[
        "calibrated_quality"
    ] = calibrator.predict(
        binary_groups[
            "good_probability"
        ]
    )


    print("\nHoldout group averages:")


    for label in [
        "bad",
        "good"
    ]:

        subset = binary_groups[
            binary_groups[
                "label"
            ] == label
        ]


        print(
            f"\n{label.upper()}"
        )

        print(
            f"  Raw probability mean: "
            f"{subset['good_probability'].mean():.3f}"
        )

        print(
            f"  Calibrated score mean: "
            f"{subset['calibrated_quality'].mean() * 100:.1f}"
        )


    print(
        "\nCalibrator saved to:"
    )

    print(
        CALIBRATOR_PATH
    )


if __name__ == "__main__":
    main()