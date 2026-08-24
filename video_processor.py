import os
import csv
import cv2
import mediapipe as mp

from feature_extractor import extract_features


# =========================================================
# PATHS
# =========================================================

REAL_VIDEO_ROOT = "data/real_videos"
SYNTHETIC_VIDEO_ROOT = "data/synthetic_videos"

REAL_OUTPUT_CSV = (
    "data/processed/real_video_features.csv"
)

SYNTHETIC_OUTPUT_CSV = (
    "data/processed/synthetic_good_bad_features.csv"
)

BORDERLINE_OUTPUT_CSV = (
    "data/processed/synthetic_borderline_features.csv"
)

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
FACE_MODEL_PATH = "models/face_landmarker.task"


# =========================================================
# SETTINGS
# =========================================================

# Synthetic clips are short and repetitive,
# so one frame per second is enough.
SYNTHETIC_SAMPLE_INTERVAL = 1.0

# Real videos contain more natural variation.
# 1 frame every 1.5 seconds avoids near-duplicates.
REAL_SAMPLE_INTERVAL = 1.5


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode


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


pose_detector = PoseLandmarker.create_from_options(
    pose_options
)

face_detector = FaceLandmarker.create_from_options(
    face_options
)


# =========================================================
# HELPERS
# =========================================================

def landmarks_to_list(landmarks):
    return [
        [
            landmark.x,
            landmark.y,
            landmark.z
        ]
        for landmark in landmarks
    ]


def extract_frame_features(frame):
    """
    Pose is required.
    Face is optional.
    """

    if frame is None or frame.size == 0:
        return None, "invalid_frame"

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    pose_result = pose_detector.detect(
        mp_image
    )

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


def write_csv(path, rows):
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
        writer.writerows(rows)


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def process_video(
    video_path,
    label,
    source_type,
    sample_interval,
    rows
):
    """
    Sample frames from one video and extract features.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(
            f"Could not open: {video_path}"
        )
        return {
            "successful": 0,
            "failed": 0,
            "pose_and_face": 0,
            "pose_only": 0
        }

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_interval = max(
        1,
        int(fps * sample_interval)
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    successful = 0
    failed = 0
    pose_and_face = 0
    pose_only = 0

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        features, status = (
            extract_frame_features(
                frame
            )
        )

        timestamp_seconds = (
            frame_index / fps
        )

        if features is None:
            failed += 1

            frame_index += 1
            continue

        row = features.copy()

        row["label"] = label
        row["source_type"] = source_type
        row["source_file"] = os.path.basename(
            video_path
        )
        row["frame_index"] = frame_index
        row["timestamp_seconds"] = round(
            timestamp_seconds,
            2
        )

        rows.append(row)

        successful += 1

        if status == "pose_and_face":
            pose_and_face += 1

        elif status == "pose_only":
            pose_only += 1

        frame_index += 1

    cap.release()

    print(
        f"    Total frames: {total_frames}"
    )
    print(
        f"    Extracted: {successful}"
    )
    print(
        f"      Pose + Face: {pose_and_face}"
    )
    print(
        f"      Pose only:   {pose_only}"
    )
    print(
        f"    Failed sampled frames: {failed}"
    )

    return {
        "successful": successful,
        "failed": failed,
        "pose_and_face": pose_and_face,
        "pose_only": pose_only
    }


# =========================================================
# PROCESS FOLDER
# =========================================================

def process_folder(
    folder_path,
    label,
    source_type,
    sample_interval,
    rows
):
    """
    Process all supported videos in one folder.
    """

    if not os.path.exists(
        folder_path
    ):
        return

    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )

    for filename in sorted(
        os.listdir(
            folder_path
        )
    ):

        if not filename.lower().endswith(
            video_extensions
        ):
            continue

        video_path = os.path.join(
            folder_path,
            filename
        )

        print(
            f"\nProcessing {filename}"
        )

        process_video(
            video_path,
            label,
            source_type,
            sample_interval,
            rows
        )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    real_rows = []
    synthetic_rows = []
    borderline_rows = []

    # =====================================================
    # REAL VIDEOS
    # =====================================================

    print("\n==============================")
    print("REAL VIDEOS")
    print("==============================")

    process_folder(
        os.path.join(
            REAL_VIDEO_ROOT,
            "good"
        ),
        label="good",
        source_type="real",
        sample_interval=REAL_SAMPLE_INTERVAL,
        rows=real_rows
    )

    process_folder(
        os.path.join(
            REAL_VIDEO_ROOT,
            "bad"
        ),
        label="bad",
        source_type="real",
        sample_interval=REAL_SAMPLE_INTERVAL,
        rows=real_rows
    )


    # =====================================================
    # SYNTHETIC GOOD / BAD
    # =====================================================

    print("\n==============================")
    print("SYNTHETIC GOOD / BAD")
    print("==============================")

    process_folder(
        os.path.join(
            SYNTHETIC_VIDEO_ROOT,
            "good"
        ),
        label="good",
        source_type="synthetic",
        sample_interval=SYNTHETIC_SAMPLE_INTERVAL,
        rows=synthetic_rows
    )

    process_folder(
        os.path.join(
            SYNTHETIC_VIDEO_ROOT,
            "bad"
        ),
        label="bad",
        source_type="synthetic",
        sample_interval=SYNTHETIC_SAMPLE_INTERVAL,
        rows=synthetic_rows
    )


    # =====================================================
    # BORDERLINE
    # =====================================================

    print("\n==============================")
    print("SYNTHETIC BORDERLINE")
    print("==============================")

    process_folder(
        os.path.join(
            SYNTHETIC_VIDEO_ROOT,
            "borderline"
        ),
        label="borderline",
        source_type="synthetic",
        sample_interval=SYNTHETIC_SAMPLE_INTERVAL,
        rows=borderline_rows
    )


    # =====================================================
    # SAVE
    # =====================================================

    write_csv(
        REAL_OUTPUT_CSV,
        real_rows
    )

    write_csv(
        SYNTHETIC_OUTPUT_CSV,
        synthetic_rows
    )

    write_csv(
        BORDERLINE_OUTPUT_CSV,
        borderline_rows
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n==============================")
    print("FINAL VIDEO EXTRACTION SUMMARY")
    print("==============================")

    print(
        f"Real samples: {len(real_rows)}"
    )

    print(
        f"Synthetic good/bad samples: "
        f"{len(synthetic_rows)}"
    )

    print(
        f"Borderline samples: "
        f"{len(borderline_rows)}"
    )

    print()

    if real_rows:
        real_good = sum(
            row["label"] == "good"
            for row in real_rows
        )

        real_bad = sum(
            row["label"] == "bad"
            for row in real_rows
        )

        print(
            f"Real GOOD: {real_good}"
        )

        print(
            f"Real BAD:  {real_bad}"
        )


    if synthetic_rows:
        synthetic_good = sum(
            row["label"] == "good"
            for row in synthetic_rows
        )

        synthetic_bad = sum(
            row["label"] == "bad"
            for row in synthetic_rows
        )

        print(
            f"Synthetic GOOD: "
            f"{synthetic_good}"
        )

        print(
            f"Synthetic BAD:  "
            f"{synthetic_bad}"
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