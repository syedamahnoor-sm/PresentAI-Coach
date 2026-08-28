import cv2
import time
import mediapipe as mp

from feature_extractor import extract_features
from posture_scorer import PostureScorer
from gesture_analyzer import GestureAnalyzer
from gaze_analyzer import GazeAnalyzer
from session_manager import SessionManager


# =========================================================
# 1. MEDIAPIPE TASKS SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode


# =========================================================
# MODEL PATHS
# =========================================================

POSE_MODEL_PATH = (
    "models/pose_landmarker_lite.task"
)

FACE_MODEL_PATH = (
    "models/face_landmarker.task"
)


# =========================================================
# POSE DETECTOR CONFIGURATION
# =========================================================

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================================================
# FACE DETECTOR CONFIGURATION
# =========================================================

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================================================
# CREATE DETECTORS
# =========================================================

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
# CREATE ANALYZERS
# =========================================================

posture_scorer = PostureScorer(
    model_path="models/posture_classifier.pkl"
)

gesture_analyzer = GestureAnalyzer()

gaze_analyzer = GazeAnalyzer()


# =========================================================
# CREATE SESSION MANAGER
# =========================================================

session_manager = SessionManager(
    sample_interval=1.0
)

session_manager.start_session()


# =========================================================
# 2. OPEN WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# =========================================================
# 3. POSE CONNECTIONS
# =========================================================

POSE_CONNECTIONS = [

    (0, 11),
    (0, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),
]


# =========================================================
# 4. LANDMARK SMOOTHING
# =========================================================

SMOOTHING_ALPHA = 0.25


smoothed_landmarks = None

smoothed_world_landmarks = None


def smooth_landmark_list(
    raw_landmarks,
    previous_landmarks,
    alpha
):
    """
    Smooth MediaPipe landmarks using
    Exponential Moving Average.
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


    for i, landmark in enumerate(
        raw_landmarks
    ):

        previous_landmarks[i][0] = (

            alpha
            * landmark.x

            + (
                1 - alpha
            )
            * previous_landmarks[i][0]
        )


        previous_landmarks[i][1] = (

            alpha
            * landmark.y

            + (
                1 - alpha
            )
            * previous_landmarks[i][1]
        )


        previous_landmarks[i][2] = (

            alpha
            * landmark.z

            + (
                1 - alpha
            )
            * previous_landmarks[i][2]
        )


    return previous_landmarks


# =========================================================
# 5. MAIN WEBCAM LOOP
# =========================================================

while cap.isOpened():

    success, frame = cap.read()


    if not success:

        print(
            "Could not access webcam."
        )

        break


    # -----------------------------------------------------
    # Results for this frame
    # -----------------------------------------------------

    posture_result = None
    gesture_result = None
    gaze_result = None


    # -----------------------------------------------------
    # Mirror image
    # -----------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # -----------------------------------------------------
    # Convert BGR -> RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # Convert NumPy image -> MediaPipe Image
    # -----------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------------------------------
    # VIDEO mode requires increasing timestamp
    # -----------------------------------------------------

    timestamp_ms = (
        time.monotonic_ns()
        // 1_000_000
    )


    # =====================================================
    # RUN MEDIAPIPE
    # =====================================================

    pose_results = (
        pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    face_results = (
        face_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    # =====================================================
    # PROCESS POSE
    # =====================================================

    if (
        pose_results.pose_landmarks
        and pose_results.pose_world_landmarks
    ):

        raw_landmarks = (
            pose_results.pose_landmarks[0]
        )


        raw_world_landmarks = (
            pose_results.pose_world_landmarks[0]
        )


        height, width, _ = (
            frame.shape
        )


        # -------------------------------------------------
        # Smooth image landmarks
        # -------------------------------------------------

        smoothed_landmarks = (
            smooth_landmark_list(

                raw_landmarks,

                smoothed_landmarks,

                SMOOTHING_ALPHA
            )
        )


        # -------------------------------------------------
        # Smooth world landmarks
        # -------------------------------------------------

        smoothed_world_landmarks = (
            smooth_landmark_list(

                raw_world_landmarks,

                smoothed_world_landmarks,

                SMOOTHING_ALPHA
            )
        )


        # =================================================
        # FACE IS OPTIONAL
        # =================================================

        face_landmarks = None


        if face_results.face_landmarks:

            face_landmarks = (
                face_results.face_landmarks[0]
            )


        # =================================================
        # EXTRACT ML FEATURES
        # =================================================

        features = extract_features(

            smoothed_landmarks,

            smoothed_world_landmarks,

            face_landmarks
        )


        # =================================================
        # ML POSTURE SCORING
        # =================================================

        if features is not None:

            posture_result = (
                posture_scorer.update(
                    features
                )
            )


            display_score = (
                posture_result["score"]
            )


            display_status = (
                posture_result["status"]
            )


            # ---------------------------------------------
            # Display color
            # ---------------------------------------------

            if (
                display_status
                == "Excellent"
            ):

                score_color = (
                    0,
                    255,
                    0
                )


            elif (
                display_status
                == "Good"
            ):

                score_color = (
                    0,
                    220,
                    0
                )


            elif (
                display_status
                == "Needs Improvement"
            ):

                score_color = (
                    0,
                    200,
                    255
                )


            else:

                score_color = (
                    0,
                    0,
                    255
                )


            # =================================================
            # GESTURE ANALYSIS
            # =================================================

            gesture_result = (
                gesture_analyzer.update(
                    smoothed_landmarks
                )
            )


            if gesture_result is not None:

                cv2.putText(
                    frame,
                    (
                        f"Gestures: "
                        f"{gesture_result['status']} "
                        f"({gesture_result['score']}/100)"
                    ),
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )


                if gesture_result[
                    "feedback"
                ]:

                    cv2.putText(
                        frame,
                        gesture_result[
                            "feedback"
                        ][0],
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )


            # =================================================
            # GAZE / EYE-CONTACT ANALYSIS
            # =================================================

            gaze_result = (
                gaze_analyzer.update(
                    face_landmarks
                )
            )


            if gaze_result is not None:

                cv2.putText(
                    frame,
                    (
                        f"Eye Contact: "
                        f"{gaze_result['status']} "
                        f"({gaze_result['score']}/100)"
                    ),
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 200, 0),
                    2,
                )


                cv2.putText(
                    frame,
                    gaze_result[
                        "feedback"
                    ],
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )


            # =================================================
            # SESSION MANAGER
            # =================================================

            session_manager.update_visual(
                posture_result=posture_result,
                gesture_result=gesture_result,
                gaze_result=gaze_result,
            )


            # =================================================
            # MAIN POSTURE SCORE
            # =================================================

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

                score_color,

                2
            )


            # =================================================
            # ML CONFIDENCE
            # =================================================

            cv2.putText(

                frame,

                (
                    f"ML Good Confidence: "
                    f"{posture_result['good_probability']:.2f}"
                ),

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 255),

                1
            )


        # =================================================
        # DRAW POSE LANDMARKS
        # =================================================

        for landmark in smoothed_landmarks:

            x = int(
                landmark[0]
                * width
            )

            y = int(
                landmark[1]
                * height
            )


            cv2.circle(

                frame,

                (x, y),

                4,

                (0, 255, 0),

                -1
            )


        # =================================================
        # DRAW POSE SKELETON
        # =================================================

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

                int(
                    start[0]
                    * width
                ),

                int(
                    start[1]
                    * height
                )
            )


            end_point = (

                int(
                    end[0]
                    * width
                ),

                int(
                    end[1]
                    * height
                )
            )


            cv2.line(

                frame,

                start_point,

                end_point,

                (255, 255, 255),

                2
            )


        # =================================================
        # OPTIONAL FACE LANDMARK DRAWING
        # =================================================

        if face_landmarks is not None:

            important_face_points = [

                1,
                10,
                152,
                33,
                263,
            ]


            for index in (
                important_face_points
            ):

                landmark = (
                    face_landmarks[
                        index
                    ]
                )


                x = int(
                    landmark.x
                    * width
                )

                y = int(
                    landmark.y
                    * height
                )


                cv2.circle(

                    frame,

                    (x, y),

                    3,

                    (0, 255, 255),

                    -1
                )


    # =====================================================
    # NO PERSON DETECTED
    # =====================================================

    else:

        smoothed_landmarks = None

        smoothed_world_landmarks = None


        # Clear score history so an old score
        # does not carry over when person returns.
        posture_scorer.reset()

        gesture_analyzer.reset()

        gaze_analyzer.reset()


        cv2.putText(

            frame,

            "No pose detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2
        )


    # =====================================================
    # 6. DISPLAY WINDOW
    # =====================================================

    cv2.imshow(

        "PresentAI Coach - Pose Detection",

        frame
    )


    # Press Q to quit
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# =========================================================
# 7. END VISUAL SESSION
# =========================================================

visual_report = (
    session_manager.end_session()
)


print()
print(
    "===================================="
)
print(
    "VISUAL SESSION SUMMARY"
)
print(
    "===================================="
)


print(
    f"Posture: "
    f"{visual_report['dimension_scores'].get('posture')}"
)

print(
    f"Gestures: "
    f"{visual_report['dimension_scores'].get('gesture')}"
)

print(
    f"Eye Contact: "
    f"{visual_report['dimension_scores'].get('gaze')}"
)


print()
print(
    "Visual Samples:"
)

print(
    visual_report[
        "visual_samples"
    ]
)


# =========================================================
# 8. CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

pose_detector.close()

face_detector.close()