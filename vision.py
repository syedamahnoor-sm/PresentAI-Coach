import cv2
import time
import mediapipe as mp

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

# Stores smoothed landmark positions from the previous frame
smoothed_landmarks = None

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
    timestamp_ms = int(time.time() * 1000)

    # Run pose detection
    results = pose_detector.detect_for_video(mp_image, timestamp_ms)

    # -----------------------------
    # 5. Smooth and draw detected landmarks
    # -----------------------------

    if results.pose_landmarks:

        raw_landmarks = results.pose_landmarks[0]

        height, width, _ = frame.shape

        # First detected frame:
        # We don't have previous coordinates to average with yet.
        if smoothed_landmarks is None:

            smoothed_landmarks = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in raw_landmarks
            ]

        else:

            # Smooth every landmark using Exponential Moving Average
            for i, landmark in enumerate(raw_landmarks):

                smoothed_landmarks[i][0] = (
                    SMOOTHING_ALPHA * landmark.x
                    + (1 - SMOOTHING_ALPHA) * smoothed_landmarks[i][0]
                )

                smoothed_landmarks[i][1] = (
                    SMOOTHING_ALPHA * landmark.y
                    + (1 - SMOOTHING_ALPHA) * smoothed_landmarks[i][1]
                )

                smoothed_landmarks[i][2] = (
                    SMOOTHING_ALPHA * landmark.z
                    + (1 - SMOOTHING_ALPHA) * smoothed_landmarks[i][2]
                )

        # Draw every smoothed landmark
        for landmark in smoothed_landmarks:

            x = int(landmark[0] * width)
            y = int(landmark[1] * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Draw skeleton using smoothed coordinates
        for start_index, end_index in POSE_CONNECTIONS:

            start = smoothed_landmarks[start_index]
            end = smoothed_landmarks[end_index]

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
        # MediaPipe temporarily lost the person
        smoothed_landmarks = None


    # -----------------------------
    # 6. Display result
    # -----------------------------

    cv2.imshow(
        "PresentAI Coach - Pose Detection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# 7. Clean up
# -----------------------------

cap.release()

cv2.destroyAllWindows()

pose_detector.close()
