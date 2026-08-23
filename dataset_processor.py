import os
import csv
import cv2
import yaml
import mediapipe as mp

from feature_extractor import extract_features


# =========================================================
# PATHS
# =========================================================

ROBOFLOW_ROOT = "data/roboflow"

OUTPUT_CSV = (
    "data/processed/"
    "roboflow_posture_features.csv"
)

FAILED_CSV = (
    "data/processed/"
    "failed_roboflow_samples.csv"
)

POSE_MODEL_PATH = (
    "models/pose_landmarker_lite.task"
)

FACE_MODEL_PATH = (
    "models/face_landmarker.task"
)


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)


pose_detector = (
    PoseLandmarker.create_from_options(
        pose_options
    )
)

face_detector = (
    FaceLandmarker.create_from_options(
        face_options
    )
)


# =========================================================
# LABEL HELPERS
# =========================================================

def normalize_class_name(name):
    """
    Convert dataset-specific class names into:
    good / bad
    """

    name = str(name).lower().strip()

    if "good" in name:
        return "good"

    if "bad" in name:
        return "bad"

    return None


def load_class_names(dataset_path):
    """
    Load class names from Roboflow data.yaml.

    Supports both:
        names: ['bad', 'good']

    and dictionary forms.
    """

    yaml_path = os.path.join(
        dataset_path,
        "data.yaml"
    )

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    names = config["names"]

    if isinstance(names, dict):

        names = [
            names[key]
            for key in sorted(
                names,
                key=lambda value: int(value)
            )
        ]

    return names


# =========================================================
# YOLO HELPERS
# =========================================================

def read_yolo_labels(label_path):
    """
    Read YOLO annotations.

    Format:
    class_id center_x center_y width height
    """

    annotations = []

    if not os.path.exists(label_path):
        return annotations


    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            values = (
                line.strip().split()
            )

            if len(values) < 5:
                continue


            try:
                class_id = int(
                    float(values[0])
                )

                x_center = float(values[1])
                y_center = float(values[2])

                box_width = float(values[3])
                box_height = float(values[4])

            except ValueError:
                continue


            annotations.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                )
            )


    return annotations


def crop_yolo_box(
    image,
    x_center,
    y_center,
    box_width,
    box_height,
    margin=0.20
):
    """
    Crop YOLO bounding box with extra context.

    We use a larger margin than before because
    MediaPipe benefits from seeing surrounding
    shoulders, hips, head and body context.
    """

    image_height, image_width = (
        image.shape[:2]
    )


    x1 = (
        x_center
        - box_width / 2
    )

    y1 = (
        y_center
        - box_height / 2
    )

    x2 = (
        x_center
        + box_width / 2
    )

    y2 = (
        y_center
        + box_height / 2
    )


    # Add proportional margin based
    # on bounding-box dimensions.
    x_margin = box_width * margin

    y_margin = box_height * margin


    x1 -= x_margin
    x2 += x_margin

    y1 -= y_margin
    y2 += y_margin


    # Clamp normalized coordinates.
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)

    x2 = min(1.0, x2)
    y2 = min(1.0, y2)


    # Convert to pixels.
    x1 = int(x1 * image_width)
    y1 = int(y1 * image_height)

    x2 = int(x2 * image_width)
    y2 = int(y2 * image_height)


    if x2 <= x1 or y2 <= y1:
        return None


    return image[
        y1:y2,
        x1:x2
    ]


# =========================================================
# MEDIAPIPE HELPERS
# =========================================================

def landmarks_to_list(landmarks):
    """
    Convert MediaPipe landmarks into:
    [[x, y, z], ...]
    """

    return [
        [
            landmark.x,
            landmark.y,
            landmark.z
        ]
        for landmark in landmarks
    ]


def run_mediapipe(image):
    """
    Run Pose + Face on one image.

    Pose is REQUIRED.
    Face is OPTIONAL.

    Returns:
        features,
        status
    """

    if image is None:
        return None, "invalid_image"

    if image.size == 0:
        return None, "empty_image"


    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )


    pose_result = pose_detector.detect(
        mp_image
    )


    # -----------------------------------------
    # Pose is mandatory
    # -----------------------------------------

    if not pose_result.pose_landmarks:
        return None, "no_pose"


    if not pose_result.pose_world_landmarks:
        return None, "no_world_pose"


    pose_landmarks = landmarks_to_list(
        pose_result.pose_landmarks[0]
    )


    world_landmarks = landmarks_to_list(
        pose_result.pose_world_landmarks[0]
    )


    # -----------------------------------------
    # Face is optional
    # -----------------------------------------

    face_result = face_detector.detect(
        mp_image
    )


    face_landmarks = None


    if face_result.face_landmarks:

        face_landmarks = (
            face_result.face_landmarks[0]
        )


    features = extract_features(
        pose_landmarks,
        world_landmarks,
        face_landmarks
    )


    if features is None:
        return None, "feature_error"


    if face_landmarks is None:
        return features, "pose_only"


    return features, "pose_and_face"


# =========================================================
# FEATURE EXTRACTION WITH FALLBACK
# =========================================================

def extract_features_with_fallback(
    image,
    person_crop
):
    """
    First try the YOLO person crop.

    If Pose fails, retry using the entire image.

    This helps side views or imperfect bounding boxes.
    """

    crop_features = None
    crop_status = "no_crop"


    if (
        person_crop is not None
        and person_crop.size > 0
    ):

        crop_features, crop_status = (
            run_mediapipe(
                person_crop
            )
        )


    # If pose succeeded on crop,
    # keep the crop result.
    if crop_features is not None:

        return (
            crop_features,
            crop_status,
            "crop"
        )


    # -----------------------------------------
    # FALLBACK: entire image
    # -----------------------------------------

    full_features, full_status = (
        run_mediapipe(
            image
        )
    )


    if full_features is not None:

        return (
            full_features,
            full_status,
            "full_image_fallback"
        )


    # Both failed.
    return (
        None,
        full_status,
        "failed"
    )


# =========================================================
# PROCESS ONE DATASET
# =========================================================

def process_dataset(
    dataset_name,
    dataset_path,
    rows,
    failed_rows
):
    """
    Process one Roboflow dataset.
    """

    print(
        f"\nProcessing {dataset_name}..."
    )


    class_names = load_class_names(
        dataset_path
    )


    print(
        "Classes:",
        class_names
    )


    stats = {
        "successful": 0,
        "failed": 0,
        "pose_and_face": 0,
        "pose_only": 0,
        "fallback_success": 0,
        "missing_annotations": 0,
    }


    for split in [
        "train",
        "valid",
        "test"
    ]:

        images_folder = os.path.join(
            dataset_path,
            split,
            "images"
        )


        labels_folder = os.path.join(
            dataset_path,
            split,
            "labels"
        )


        if not os.path.exists(
            images_folder
        ):
            continue


        print(
            f"  Processing {split}..."
        )


        for filename in sorted(
            os.listdir(
                images_folder
            )
        ):

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                )
            ):
                continue


            image_path = os.path.join(
                images_folder,
                filename
            )


            label_filename = (
                os.path.splitext(
                    filename
                )[0]
                + ".txt"
            )


            label_path = os.path.join(
                labels_folder,
                label_filename
            )


            image = cv2.imread(
                image_path
            )


            if image is None:

                stats["failed"] += 1

                failed_rows.append({
                    "source_dataset":
                        dataset_name,

                    "split":
                        split,

                    "source_file":
                        filename,

                    "reason":
                        "image_read_failed"
                })

                continue


            annotations = read_yolo_labels(
                label_path
            )


            if not annotations:

                stats[
                    "missing_annotations"
                ] += 1

                failed_rows.append({
                    "source_dataset":
                        dataset_name,

                    "split":
                        split,

                    "source_file":
                        filename,

                    "reason":
                        "no_annotation"
                })

                continue


            for (
                box_index,
                annotation
            ) in enumerate(
                annotations
            ):

                (
                    class_id,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                ) = annotation


                if (
                    class_id < 0
                    or class_id
                    >= len(class_names)
                ):
                    continue


                original_class = (
                    class_names[
                        class_id
                    ]
                )


                label = normalize_class_name(
                    original_class
                )


                if label is None:
                    continue


                person_crop = crop_yolo_box(
                    image,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                )


                (
                    features,
                    detection_status,
                    extraction_source
                ) = (
                    extract_features_with_fallback(
                        image,
                        person_crop
                    )
                )


                if features is None:

                    stats["failed"] += 1

                    failed_rows.append({
                        "source_dataset":
                            dataset_name,

                        "split":
                            split,

                        "source_file":
                            filename,

                        "box_index":
                            box_index,

                        "reason":
                            detection_status
                    })

                    continue


                # -----------------------------------------
                # Build CSV row
                # -----------------------------------------

                row = features.copy()


                row["label"] = label

                row["source_dataset"] = (
                    dataset_name
                )

                row["original_split"] = (
                    split
                )

                row["source_file"] = (
                    filename
                )

                row["box_index"] = (
                    box_index
                )

                row["extraction_source"] = (
                    extraction_source
                )


                rows.append(row)


                # -----------------------------------------
                # Stats
                # -----------------------------------------

                stats["successful"] += 1


                if detection_status == (
                    "pose_and_face"
                ):

                    stats[
                        "pose_and_face"
                    ] += 1


                elif detection_status == (
                    "pose_only"
                ):

                    stats[
                        "pose_only"
                    ] += 1


                if extraction_source == (
                    "full_image_fallback"
                ):

                    stats[
                        "fallback_success"
                    ] += 1


    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    print(
        f"  Successful samples: "
        f"{stats['successful']}"
    )

    print(
        f"    Pose + Face: "
        f"{stats['pose_and_face']}"
    )

    print(
        f"    Pose only:   "
        f"{stats['pose_only']}"
    )

    print(
        f"    Full-image fallback rescued: "
        f"{stats['fallback_success']}"
    )

    print(
        f"  Failed samples: "
        f"{stats['failed']}"
    )

    print(
        f"  Missing annotations: "
        f"{stats['missing_annotations']}"
    )


    return stats


# =========================================================
# WRITE CSV
# =========================================================

def write_csv(
    path,
    rows
):
    """
    Write rows with a stable union of all columns.
    """

    if not rows:
        return


    fieldnames = []

    seen = set()


    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(key)

                fieldnames.append(key)


    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =========================================================
# MAIN
# =========================================================

def main():
    """
    Process all three Roboflow datasets.
    """

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    rows = []

    failed_rows = []


    datasets = [
        "dataset_1",
        "dataset_2",
        "dataset_3"
    ]


    total_stats = {
        "successful": 0,
        "failed": 0,
        "pose_and_face": 0,
        "pose_only": 0,
        "fallback_success": 0,
    }


    for dataset_name in datasets:

        dataset_path = os.path.join(
            ROBOFLOW_ROOT,
            dataset_name
        )


        if not os.path.exists(
            dataset_path
        ):

            print(
                f"Skipping missing dataset: "
                f"{dataset_name}"
            )

            continue


        stats = process_dataset(
            dataset_name,
            dataset_path,
            rows,
            failed_rows
        )


        for key in total_stats:

            total_stats[key] += (
                stats.get(
                    key,
                    0
                )
            )


    # =====================================================
    # NO DATA
    # =====================================================

    if not rows:

        print(
            "\nNo usable samples were extracted."
        )

        return


    # =====================================================
    # SAVE FEATURES
    # =====================================================

    write_csv(
        OUTPUT_CSV,
        rows
    )


    # =====================================================
    # SAVE FAILED SAMPLE LOG
    # =====================================================

    if failed_rows:

        write_csv(
            FAILED_CSV,
            failed_rows
        )


    # =====================================================
    # CLASS COUNTS
    # =====================================================

    good_count = sum(
        row["label"] == "good"
        for row in rows
    )

    bad_count = sum(
        row["label"] == "bad"
        for row in rows
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 55)

    print("FINAL EXTRACTION SUMMARY")

    print("=" * 55)


    print(
        f"Total usable samples: "
        f"{len(rows)}"
    )

    print(
        f"GOOD: {good_count}"
    )

    print(
        f"BAD:  {bad_count}"
    )


    print()

    print(
        f"Pose + Face samples: "
        f"{total_stats['pose_and_face']}"
    )

    print(
        f"Pose-only samples:   "
        f"{total_stats['pose_only']}"
    )

    print(
        f"Full-image fallback rescued: "
        f"{total_stats['fallback_success']}"
    )

    print(
        f"Truly failed samples: "
        f"{total_stats['failed']}"
    )


    print()

    print(
        "Features saved to:"
    )

    print(
        OUTPUT_CSV
    )


    if failed_rows:

        print()

        print(
            "Failure log saved to:"
        )

        print(
            FAILED_CSV
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        main()

    finally:

        pose_detector.close()

        face_detector.close()