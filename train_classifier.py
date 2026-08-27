import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC


# =========================================================
# PATHS
# =========================================================

ROBOFLOW_CSV = (
    "data/processed/"
    "roboflow_posture_features.csv"
)

REAL_VIDEO_CSV = (
    "data/processed/"
    "real_video_features.csv"
)

SYNTHETIC_CSV = (
    "data/processed/"
    "synthetic_good_bad_features.csv"
)

BORDERLINE_CSV = (
    "data/processed/"
    "synthetic_borderline_features.csv"
)


MODEL_DIR = "models"

EVALUATION_MODEL_PATH = (
    "models/posture_classifier_eval.pkl"
)

FINAL_MODEL_PATH = (
    "models/posture_classifier.pkl"
)

CONFUSION_MATRIX_PATH = (
    "data/processed/"
    "posture_confusion_matrix.png"
)

FEATURE_LIST_PATH = (
    "data/processed/"
    "posture_feature_names.txt"
)


# =========================================================
# RANDOM SEED
# =========================================================

RANDOM_STATE = 42


# =========================================================
# NON-FEATURE COLUMNS
# =========================================================

METADATA_COLUMNS = {
    "label",

    "source_dataset",
    "source_type",
    "source_file",

    "original_split",
    "box_index",
    "extraction_source",

    "frame_index",
    "timestamp_seconds",

    "group_id",
}


# =========================================================
# LOAD DATA
# =========================================================

def load_csv(path, source_name):
    """
    Load one processed feature CSV.

    Adds a dataset-source column so we know
    where every sample originated.
    """

    if not os.path.exists(path):

        print(
            f"WARNING: file not found: {path}"
        )

        return pd.DataFrame()


    dataframe = pd.read_csv(path)

    dataframe["data_source"] = (
        source_name
    )

    return dataframe


# =========================================================
# CREATE LEAKAGE-SAFE GROUPS
# =========================================================

def create_group_id(row):
    """
    Create a grouping key.

    All frames from the same video must remain
    together during train/test splitting.

    Roboflow still images are grouped by
    dataset + source image.
    """

    source_file = str(
        row.get(
            "source_file",
            "unknown"
        )
    )


    # -----------------------------------------
    # VIDEO DATA
    # -----------------------------------------

    if row["data_source"] in {
        "real_video",
        "synthetic_video"
    }:

        return (
            f"{row['data_source']}::"
            f"{source_file}"
        )


    # -----------------------------------------
    # ROBOFLOW IMAGE DATA
    # -----------------------------------------

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


# =========================================================
# BUILD DATASET
# =========================================================

def build_dataset():

    roboflow_df = load_csv(
        ROBOFLOW_CSV,
        "roboflow"
    )


    real_df = load_csv(
        REAL_VIDEO_CSV,
        "real_video"
    )


    synthetic_df = load_csv(
        SYNTHETIC_CSV,
        "synthetic_video"
    )


    dataframes = [
        dataframe
        for dataframe in [
            roboflow_df,
            real_df,
            synthetic_df,
        ]
        if not dataframe.empty
    ]


    if not dataframes:

        raise RuntimeError(
            "No training feature CSV files found."
        )


    data = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False
    )


    # -----------------------------------------
    # Keep only GOOD / BAD
    # -----------------------------------------

    data["label"] = (
        data["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )


    data = data[
        data["label"].isin(
            ["good", "bad"]
        )
    ].copy()


    # -----------------------------------------
    # Create group IDs
    # -----------------------------------------

    data["group_id"] = data.apply(
        create_group_id,
        axis=1
    )


    return data


# =========================================================
# SELECT FEATURE COLUMNS
# =========================================================

def get_feature_columns(data):
    """
    Use only numerical posture features.

    Metadata such as filenames, labels and
    timestamps must NOT enter the classifier.
    """

    candidate_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
        and column != "data_source"
    ]


    numeric_columns = []

    for column in candidate_columns:

        converted = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        # If at least one valid numeric value exists,
        # treat it as an ML feature.
        if converted.notna().any():

            data[column] = converted

            numeric_columns.append(
                column
            )


    return numeric_columns


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def build_models():
    """
    Build several suitable models.

    We compare models instead of assuming
    Random Forest is automatically best.
    """

    models = {}


    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    models["Logistic Regression"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,

                    max_depth=None,

                    min_samples_leaf=2,

                    class_weight="balanced",

                    random_state=RANDOM_STATE,

                    n_jobs=-1,
                )
            ),
        ]
    )


    # =====================================================
    # RBF SVM
    # =====================================================

    models["RBF SVM"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",

                    C=2.0,

                    gamma="scale",

                    class_weight="balanced",

                    probability=True,

                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )


    return models


# =========================================================
# CREATE HOLDOUT SPLIT
# =========================================================

def create_holdout_split(
    data,
    test_fraction=0.20
):
    """
    Create a source-aware, label-aware,
    group-safe holdout split.

    Ensures good and bad examples from each
    data source are represented in the holdout.

    Entire videos/groups stay together.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    test_groups = set()


    # Split independently within every
    # source + label combination
    for (
        data_source,
        label
    ), subset in data.groupby(
        [
            "data_source",
            "label"
        ]
    ):

        unique_groups = (
            subset[
                "group_id"
            ]
            .drop_duplicates()
            .tolist()
        )


        rng.shuffle(
            unique_groups
        )


        # At least one group from every
        # source/label combination
        number_test_groups = max(
            1,
            round(
                len(unique_groups)
                * test_fraction
            )
        )


        selected_groups = (
            unique_groups[
                :number_test_groups
            ]
        )


        test_groups.update(
            selected_groups
        )


    test_mask = (
        data[
            "group_id"
        ].isin(
            test_groups
        )
    )


    test_indices = np.where(
        test_mask
    )[0]


    train_indices = np.where(
        ~test_mask
    )[0]


    return (
        train_indices,
        test_indices
    )

# =========================================================
# CROSS-VALIDATE MODELS
# =========================================================

def compare_models(
    models,
    X_train,
    y_train,
    train_groups
):
    """
    Compare models using group-aware CV
    on TRAINING DATA ONLY.

    Holdout test data remains untouched.
    """

    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    results = {}


    print("\n" + "=" * 60)

    print(
        "GROUP-AWARE CROSS-VALIDATION"
    )

    print("=" * 60)


    for name, model in models.items():

        scores = cross_val_score(
            model,

            X_train,
            y_train,

            groups=train_groups,

            cv=cv,

            scoring="f1_macro",

            n_jobs=-1,
        )


        mean_score = scores.mean()

        std_score = scores.std()


        results[name] = {
            "model": model,
            "mean_f1": mean_score,
            "std_f1": std_score,
        }


        print(
            f"\n{name}"
        )

        print(
            f"  Fold F1 scores: "
            f"{np.round(scores, 4)}"
        )

        print(
            f"  Mean Macro F1: "
            f"{mean_score:.4f}"
        )

        print(
            f"  Std: "
            f"{std_score:.4f}"
        )


    return results


# =========================================================
# SELECT BEST MODEL
# =========================================================

def choose_best_model(results):

    best_name = max(
        results,
        key=lambda name:
        results[name]["mean_f1"]
    )


    best_model = (
        results[
            best_name
        ]["model"]
    )


    return (
        best_name,
        best_model
    )


# =========================================================
# EVALUATE HOLDOUT
# =========================================================

def evaluate_holdout(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)

    print(
        "UNSEEN GROUP-AWARE HOLDOUT TEST"
    )

    print("=" * 60)


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        pos_label="good",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        pos_label="good",
        zero_division=0
    )


    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision (GOOD): "
        f"{precision:.4f}"
    )

    print(
        f"Recall (GOOD): "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )


    # -----------------------------------------
    # Probability-based metrics
    # -----------------------------------------

    roc_auc = None


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X_test
            )
        )


        class_names = list(
            model.classes_
        )


        if (
            "good" in class_names
            and len(
                np.unique(y_test)
            ) == 2
        ):

            good_index = (
                class_names.index(
                    "good"
                )
            )


            good_probabilities = (
                probabilities[
                    :,
                    good_index
                ]
            )


            binary_truth = (
                y_test == "good"
            ).astype(int)


            roc_auc = roc_auc_score(
                binary_truth,
                good_probabilities
            )


            print(
                f"ROC-AUC: "
                f"{roc_auc:.4f}"
            )


    # -----------------------------------------
    # Classification report
    # -----------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )


    # -----------------------------------------
    # Confusion matrix
    # -----------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "bad",
            "good"
        ]
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Bad",
            "Good"
        ]
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )


    ax.set_title(
        "PresentAI Posture Classifier\n"
        "Group-Aware Holdout Test"
    )


    plt.tight_layout()


    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=180
    )


    plt.close(fig)


    print(
        "Confusion matrix saved to:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
    }


# =========================================================
# PRINT DATASET SUMMARY
# =========================================================

def print_dataset_summary(
    data,
    train_indices,
    test_indices
):

    print("\n" + "=" * 60)

    print(
        "DATASET SUMMARY"
    )

    print("=" * 60)


    print(
        f"Total rows: {len(data)}"
    )


    print(
        "\nOverall labels:"
    )

    print(
        data[
            "label"
        ].value_counts()
    )


    print(
        "\nBy source:"
    )

    print(
        pd.crosstab(
            data["data_source"],
            data["label"]
        )
    )


    print(
        "\nUnique groups:"
    )

    print(
        data[
            "group_id"
        ].nunique()
    )


    train_data = data.iloc[
        train_indices
    ]


    test_data = data.iloc[
        test_indices
    ]


    print(
        "\nTRAIN:"
    )

    print(
        f"Rows: "
        f"{len(train_data)}"
    )

    print(
        f"Groups: "
        f"{train_data['group_id'].nunique()}"
    )

    print(
        train_data[
            "label"
        ].value_counts()
    )


    print(
        "\nHOLDOUT TEST:"
    )

    print(
        f"Rows: "
        f"{len(test_data)}"
    )

    print(
        f"Groups: "
        f"{test_data['group_id'].nunique()}"
    )

    print(
        test_data[
            "label"
        ].value_counts()
    )


    # -----------------------------------------
    # Confirm zero group leakage
    # -----------------------------------------

    train_groups = set(
        train_data[
            "group_id"
        ]
    )


    test_groups = set(
        test_data[
            "group_id"
        ]
    )


    overlap = (
        train_groups
        & test_groups
    )


    print(
        "\nGroup leakage check:"
    )

    print(
        f"Overlapping groups: "
        f"{len(overlap)}"
    )


    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "same group exists in "
            "train and test."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    # =====================================================
    # LOAD
    # =====================================================

    data = build_dataset()


    # =====================================================
    # FEATURES
    # =====================================================

    feature_columns = (
        get_feature_columns(
            data
        )
    )


    print(
        "\nFeature columns:"
    )


    for feature in feature_columns:

        print(
            f"  - {feature}"
        )


    with open(
        FEATURE_LIST_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for feature in feature_columns:

            file.write(
                feature + "\n"
            )


    X = data[
        feature_columns
    ].copy()


    y = data[
        "label"
    ].copy()


    groups = data[
        "group_id"
    ].copy()


    # =====================================================
    # HOLDOUT SPLIT
    # =====================================================

    (
        train_indices,
        test_indices
    ) = create_holdout_split(
        data
    )


    print_dataset_summary(
        data,
        train_indices,
        test_indices
    )


    X_train = X.iloc[
        train_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    train_groups = groups.iloc[
        train_indices
    ]


    X_test = X.iloc[
        test_indices
    ]

    y_test = y.iloc[
        test_indices
    ]


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    models = build_models()


    results = compare_models(
        models,
        X_train,
        y_train,
        train_groups
    )


    # =====================================================
    # BEST MODEL
    # =====================================================

    (
        best_name,
        best_model
    ) = choose_best_model(
        results
    )


    print("\n" + "=" * 60)

    print(
        f"BEST CV MODEL: "
        f"{best_name}"
    )

    print("=" * 60)


    # =====================================================
    # HOLDOUT EVALUATION
    # =====================================================

    evaluation_model = clone(
        best_model
    )


    metrics = evaluate_holdout(
        evaluation_model,
        X_train,
        y_train,
        X_test,
        y_test
    )


    # =====================================================
    # SAVE EVALUATION MODEL
    # =====================================================

    evaluation_package = {
        "model":
            evaluation_model,

        "features":
            feature_columns,

        "model_name":
            best_name,

        "metrics":
            metrics,
        
        "holdout_groups": list(
            groups.iloc[test_indices].unique()
        ),
    }


    joblib.dump(
        evaluation_package,
        EVALUATION_MODEL_PATH
    )


    print(
        "\nEvaluation model saved to:"
    )

    print(
        EVALUATION_MODEL_PATH
    )


    # =====================================================
    # FINAL DEPLOYMENT MODEL
    # =====================================================
    #
    # Only AFTER honest holdout evaluation,
    # retrain the chosen model on all labeled
    # good/bad data for use in PresentAI.
    # =====================================================

    final_model = clone(
        best_model
    )


    final_model.fit(
        X,
        y
    )


    final_package = {
        "model":
            final_model,

        "features":
            feature_columns,

        "model_name":
            best_name,

        "evaluation_metrics":
            metrics,
    }


    joblib.dump(
        final_package,
        FINAL_MODEL_PATH
    )


    print(
        "\nFinal deployment model saved to:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print("\n" + "=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()