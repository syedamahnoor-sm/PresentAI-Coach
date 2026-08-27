import os
import pandas as pd

from posture_scorer import PostureScorer

# =========================================
# PATHS
# =========================================

ROBOFLOW_CSV = "data/processed/" "roboflow_posture_features.csv"

REAL_CSV = "data/processed/" "real_video_features.csv"

SYNTHETIC_CSV = "data/processed/" "synthetic_good_bad_features.csv"

BORDERLINE_CSV = "data/processed/" "synthetic_borderline_features.csv"


# =========================================
# HELPERS
# =========================================


def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


def row_to_features(row, feature_names):
    """
    Convert a dataframe row into the feature
    dictionary expected by PostureScorer.
    """

    features = {}

    for name in feature_names:
        features[name] = row.get(name)

    return features


def test_class_samples(name, dataframe, scorer, label=None):
    """
    Evaluate the full class distribution.
    """

    if dataframe.empty:
        return

    if label is not None:

        dataframe = dataframe[dataframe["label"] == label]

    if dataframe.empty:
        return

    raw_scores = []

    # Important:
    # each row is evaluated independently here.
    # We do NOT want temporal history from unrelated
    # images/videos affecting the distribution.
    for _, row in dataframe.iterrows():

        scorer.reset()

        features = row_to_features(row, scorer.feature_names)

        result = scorer.update(features)

        raw_scores.append(result["score"])

    series = pd.Series(raw_scores)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Samples: {len(series)}")

    print(f"Mean:   {series.mean():.2f}")

    print(f"Median: {series.median():.2f}")

    print(f"Min:    {series.min():.2f}")

    print(f"10th percentile: " f"{series.quantile(0.10):.2f}")

    print(f"25th percentile: " f"{series.quantile(0.25):.2f}")

    print(f"75th percentile: " f"{series.quantile(0.75):.2f}")

    print(f"90th percentile: " f"{series.quantile(0.90):.2f}")

    print(f"Max:    {series.max():.2f}")


def test_borderline(dataframe, scorer):
    """
    Test every borderline video separately.
    """

    if dataframe.empty:
        return

    print("\n" + "=" * 50)
    print("BORDERLINE VIDEOS")
    print("=" * 50)

    for source_file, group in dataframe.groupby("source_file"):

        scorer.reset()

        scores = []

        for _, row in group.iterrows():

            features = row_to_features(row, scorer.feature_names)

            result = scorer.update(features)

            scores.append(result["score"])

        print(f"\n{source_file}")

        print(f"  Average: " f"{sum(scores) / len(scores):.2f}")

        print(f"  Min: " f"{min(scores):.2f}")

        print(f"  Max: " f"{max(scores):.2f}")


# =========================================
# MAIN
# =========================================


def main():

    scorer = PostureScorer()

    roboflow = load_csv(ROBOFLOW_CSV)

    real = load_csv(REAL_CSV)

    synthetic = load_csv(SYNTHETIC_CSV)

    borderline = load_csv(BORDERLINE_CSV)

    test_class_samples("ROBOFLOW GOOD", roboflow, scorer, label="good")

    test_class_samples("REAL VIDEO GOOD", real, scorer, label="good")

    test_class_samples("SYNTHETIC GOOD", synthetic, scorer, label="good")

    test_class_samples("ROBOFLOW BAD", roboflow, scorer, label="bad")

    test_class_samples("REAL VIDEO BAD", real, scorer, label="bad")

    test_class_samples("SYNTHETIC BAD", synthetic, scorer, label="bad")

    # -------------------------------------
    # Borderline
    # -------------------------------------

    test_borderline(borderline, scorer)


if __name__ == "__main__":
    main()
