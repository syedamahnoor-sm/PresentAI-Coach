import cv2
import time
import mediapipe as mp
from analyzer import (
    analyze_posture,
    extract_posture_metrics,
    create_posture_baseline,
)

# -----------------------------
# 1. MediaPipe Tasks setup
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Path to the downloaded pose model
MODEL_PATH = "models/pose_landmarker_lite.task"


# Create configuration for the pose detector
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    # We are processing webcam frames continuously
    running_mode=RunningMode.VIDEO,
    # Detect only one person
    num_poses=1,
    # Confidence thresholds
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# Create pose detector
pose_detector = PoseLandmarker.create_from_options(options)


# -----------------------------
# 2. Open webcam
# -----------------------------

cap = cv2.VideoCapture(0)


# -----------------------------
# 3. Pose connections
# -----------------------------

# We define simple connections ourselves because we are no longer
# using the old mp.solutions.drawing_utils API.

POSE_CONNECTIONS = [
    (0, 11),  # nose -> left shoulder
    (0, 12),  # nose -> right shoulder
    (11, 12),  # shoulders
    (11, 13),  # left shoulder -> left elbow
    (13, 15),  # left elbow -> left wrist
    (12, 14),  # right shoulder -> right elbow
    (14, 16),  # right elbow -> right wrist
    (11, 23),  # left shoulder -> left hip
    (12, 24),  # right shoulder -> right hip
    (23, 24),  # hips
    (23, 25),  # left hip -> left knee
    (25, 27),  # left knee -> left ankle
    (24, 26),  # right hip -> right knee
    (26, 28),  # right knee -> right ankle
]

SMOOTHING_ALPHA = 0.25

# Image landmarks used for drawing
smoothed_landmarks = None

# 3D world landmarks used for posture analysis
smoothed_world_landmarks = None


# ---------------------------------
# Calibration
# ---------------------------------

CALIBRATION_SECONDS = 3

calibration_start_time = None
calibration_samples = []

posture_baseline = None


# ---------------------------------
# Score smoothing
# ---------------------------------

SCORE_SMOOTHING_ALPHA = 0.15

smoothed_posture_score = None

def smooth_landmark_list(
    raw_landmarks,
    previous_landmarks,
    alpha
):
    """
    Smooth a list of MediaPipe landmarks using EMA.
    """

    if previous_landmarks is None:

        return [
            [
                landmark.x,
                landmark.y,
                landmark.z
            ]
            for landmark in raw_landmarks
        ]


    for i, landmark in enumerate(raw_landmarks):

        previous_landmarks[i][0] = (
            alpha * landmark.x
            + (1 - alpha)
            * previous_landmarks[i][0]
        )

        previous_landmarks[i][1] = (
            alpha * landmark.y
            + (1 - alpha)
            * previous_landmarks[i][1]
        )

        previous_landmarks[i][2] = (
            alpha * landmark.z
            + (1 - alpha)
            * previous_landmarks[i][2]
        )


    return previous_landmarks

# -----------------------------
# 4. Process webcam frames
# -----------------------------

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    # Flip the image so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    # OpenCV gives us BGR images
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy image into a MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # MediaPipe VIDEO mode requires a timestamp
    timestamp_ms = time.monotonic_ns() // 1_000_000

    # Run pose detection
    results = pose_detector.detect_for_video(mp_image, timestamp_ms)

    # -----------------------------
    # 5. Process detected pose
    # -----------------------------

    if (
        results.pose_landmarks
        and results.pose_world_landmarks
    ):

        raw_landmarks = (
            results.pose_landmarks[0]
        )

        raw_world_landmarks = (
            results.pose_world_landmarks[0]
        )


        height, width, _ = frame.shape


        # ---------------------------------
        # Smooth image landmarks
        # ---------------------------------

        smoothed_landmarks = (
            smooth_landmark_list(
                raw_landmarks,
                smoothed_landmarks,
                SMOOTHING_ALPHA
            )
        )


        # ---------------------------------
        # Smooth 3D world landmarks
        # ---------------------------------

        smoothed_world_landmarks = (
            smooth_landmark_list(
                raw_world_landmarks,
                smoothed_world_landmarks,
                SMOOTHING_ALPHA
            )
        )


        # =================================
        # CALIBRATION
        # =================================

        if posture_baseline is None:

            if calibration_start_time is None:

                calibration_start_time = (
                    time.monotonic()
                )


            calibration_elapsed = (
                time.monotonic()
                - calibration_start_time
            )


            current_metrics = (
                extract_posture_metrics(
                    smoothed_landmarks,
                    smoothed_world_landmarks
                )
            )


            calibration_samples.append(
                current_metrics
            )


            remaining = max(
                0,
                CALIBRATION_SECONDS
                - calibration_elapsed
            )


            cv2.putText(
                frame,
                "Sit or stand naturally upright",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Calibrating: {remaining:.1f}s",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            if (
                calibration_elapsed
                >= CALIBRATION_SECONDS
                and len(calibration_samples) >= 20
            ):

                posture_baseline = (
                    create_posture_baseline(
                        calibration_samples
                    )
                )

                print(
                    "Posture calibration complete:"
                )

                print(posture_baseline)


        # =================================
        # POSTURE ANALYSIS
        # =================================

        else:

            posture_result = analyze_posture(
                smoothed_landmarks,
                smoothed_world_landmarks,
                posture_baseline
            )


            # ---------------------------------
            # Smooth final posture score
            # ---------------------------------

            current_score = (
                posture_result["score"]
            )


            if smoothed_posture_score is None:

                smoothed_posture_score = (
                    current_score
                )

            else:

                smoothed_posture_score = (
                    SCORE_SMOOTHING_ALPHA
                    * current_score

                    + (
                        1
                        - SCORE_SMOOTHING_ALPHA
                    )
                    * smoothed_posture_score
                )


            display_score = round(
                smoothed_posture_score,
                1
            )


            if display_score >= 90:
                display_status = "Excellent"

            elif display_score >= 75:
                display_status = "Good"

            elif display_score >= 55:
                display_status = (
                    "Needs Improvement"
                )

            else:
                display_status = "Poor"


            # ---------------------------------
            # Main score
            # ---------------------------------

            cv2.putText(
                frame,
                (
                    f"Posture: "
                    f"{display_status} "
                    f"({display_score}/100)"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )


            # ---------------------------------
            # Temporary detailed metrics
            # ---------------------------------

            cv2.putText(
                frame,
                (
                    f"Shoulders: "
                    f"{posture_result['shoulder_score']}"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                (
                    f"Sideways: "
                    f"{posture_result['sideways_score']}"
                ),
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                (
                    f"Forward: "
                    f"{posture_result['forward_score']}"
                ),
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                (
                    f"Head Tilt: "
                    f"{posture_result['head_tilt_score']}"
                ),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                (
                    f"Head Forward: "
                    f"{posture_result['head_forward_score']}"
                ),
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


        # =================================
        # DRAW LANDMARKS
        # =================================

        for landmark in smoothed_landmarks:

            x = int(
                landmark[0] * width
            )

            y = int(
                landmark[1] * height
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # Draw skeleton
        for (
            start_index,
            end_index
        ) in POSE_CONNECTIONS:

            start = (
                smoothed_landmarks[
                    start_index
                ]
            )

            end = (
                smoothed_landmarks[
                    end_index
                ]
            )

            start_point = (
                int(start[0] * width),
                int(start[1] * height)
            )

            end_point = (
                int(end[0] * width),
                int(end[1] * height)
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 255, 255),
                2
            )


    else:

        smoothed_landmarks = None
        smoothed_world_landmarks = None

    # -----------------------------
    # 6. Display result
    # -----------------------------

    cv2.imshow("PresentAI Coach - Pose Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# 7. Clean up
# -----------------------------

cap.release()

cv2.destroyAllWindows()

pose_detector.close()
