import mediapipe as mp

from feature_extractor import extract_features
from posture_scorer import PostureScorer
from gesture_analyzer import GestureAnalyzer
from gaze_analyzer import GazeAnalyzer


# =========================================================
# MODEL PATHS
# =========================================================

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
FACE_MODEL_PATH = "models/face_landmarker.task"


# =========================================================
# MEDIAPIPE TYPES
# =========================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode


# =========================================================
# SHARED VISUAL ANALYZER
# =========================================================

class VisualAnalyzer:
    """
    Shared PresentAI visual analysis engine.

    Used by:
        - webcam/live presentation mode
        - uploaded video mode

    Performs:
        - MediaPipe pose detection
        - MediaPipe face detection
        - posture scoring
        - gesture analysis
        - gaze analysis

    Performance strategy:
        - Pose every N frames
        - Face every M frames
        - Reuse latest results between inference frames
    """

    def __init__(
        self,
        pose_every_n_frames=2,
        face_every_n_frames=4,
        smoothing_alpha=0.25,
    ):
        self.pose_every_n_frames = max(
            1,
            pose_every_n_frames
        )

        self.face_every_n_frames = max(
            1,
            face_every_n_frames
        )

        self.smoothing_alpha = (
            smoothing_alpha
        )

        # -------------------------------------------------
        # MediaPipe detectors
        # -------------------------------------------------

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

        self.pose_detector = (
            PoseLandmarker.create_from_options(
                pose_options
            )
        )

        self.face_detector = (
            FaceLandmarker.create_from_options(
                face_options
            )
        )

        # -------------------------------------------------
        # PresentAI analyzers
        # -------------------------------------------------

        self.posture_scorer = PostureScorer(
            model_path=(
                "models/posture_classifier.pkl"
            )
        )

        self.gesture_analyzer = (
            GestureAnalyzer()
        )

        self.gaze_analyzer = (
            GazeAnalyzer()
        )

        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.frame_index = 0

        self.smoothed_landmarks = None
        self.smoothed_world_landmarks = None

        self.latest_face_landmarks = None

        self.latest_posture_result = None
        self.latest_gesture_result = None
        self.latest_gaze_result = None

        self.previous_pose_timestamp = -1
        self.previous_face_timestamp = -1

        self.posture_scorer.reset()
        self.gesture_analyzer.reset()
        self.gaze_analyzer.reset()


    # =====================================================
    # LANDMARK SMOOTHING
    # =====================================================

    def _smooth_landmark_list(
        self,
        raw_landmarks,
        previous_landmarks,
    ):

        alpha = (
            self.smoothing_alpha
        )

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


    # =====================================================
    # TIMESTAMP SAFETY
    # =====================================================

    @staticmethod
    def _safe_timestamp(
        timestamp_ms,
        previous_timestamp
    ):

        if timestamp_ms <= previous_timestamp:

            timestamp_ms = (
                previous_timestamp + 1
            )

        return timestamp_ms


    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process_frame(
        self,
        frame_rgb,
        timestamp_ms
    ):
        """
        Analyze one RGB frame.

        Returns a dictionary containing:
            posture_result
            gesture_result
            gaze_result
            pose_landmarks
            face_landmarks
            pose_updated
            face_updated
        """

        self.frame_index += 1

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=frame_rgb
        )


        # =================================================
        # POSE
        # =================================================

        pose_updated = False

        should_run_pose = (
            self.frame_index == 1
            or
            self.frame_index
            % self.pose_every_n_frames
            == 0
        )


        if should_run_pose:

            pose_timestamp = (
                self._safe_timestamp(
                    timestamp_ms,
                    self.previous_pose_timestamp
                )
            )

            self.previous_pose_timestamp = (
                pose_timestamp
            )

            pose_results = (
                self.pose_detector
                .detect_for_video(
                    mp_image,
                    pose_timestamp
                )
            )

            pose_updated = True


            if (
                pose_results.pose_landmarks
                and
                pose_results.pose_world_landmarks
            ):

                raw_landmarks = (
                    pose_results
                    .pose_landmarks[0]
                )

                raw_world_landmarks = (
                    pose_results
                    .pose_world_landmarks[0]
                )

                self.smoothed_landmarks = (
                    self._smooth_landmark_list(
                        raw_landmarks,
                        self.smoothed_landmarks
                    )
                )

                self.smoothed_world_landmarks = (
                    self._smooth_landmark_list(
                        raw_world_landmarks,
                        self.smoothed_world_landmarks
                    )
                )

            else:

                self.smoothed_landmarks = None
                self.smoothed_world_landmarks = None

                self.latest_posture_result = None
                self.latest_gesture_result = None

                self.posture_scorer.reset()
                self.gesture_analyzer.reset()


        # =================================================
        # FACE
        # =================================================

        face_updated = False

        should_run_face = (
            self.frame_index == 1
            or
            self.frame_index
            % self.face_every_n_frames
            == 0
        )


        if should_run_face:

            face_timestamp = (
                self._safe_timestamp(
                    timestamp_ms,
                    self.previous_face_timestamp
                )
            )

            self.previous_face_timestamp = (
                face_timestamp
            )

            face_results = (
                self.face_detector
                .detect_for_video(
                    mp_image,
                    face_timestamp
                )
            )

            face_updated = True


            if face_results.face_landmarks:

                self.latest_face_landmarks = (
                    face_results
                    .face_landmarks[0]
                )

            else:

                self.latest_face_landmarks = None

                self.latest_gaze_result = None

                self.gaze_analyzer.reset()


        # =================================================
        # POSTURE + GESTURE
        # =================================================

        if (
            pose_updated
            and
            self.smoothed_landmarks
            is not None
            and
            self.smoothed_world_landmarks
            is not None
        ):

            features = extract_features(
                self.smoothed_landmarks,
                self.smoothed_world_landmarks,
                self.latest_face_landmarks
            )


            if features is not None:

                self.latest_posture_result = (
                    self.posture_scorer.update(
                        features
                    )
                )

            else:

                self.latest_posture_result = None


            self.latest_gesture_result = (
                self.gesture_analyzer.update(
                    self.smoothed_landmarks
                )
            )


        # =================================================
        # GAZE
        # =================================================

        if face_updated:

            self.latest_gaze_result = (
                self.gaze_analyzer.update(
                    self.latest_face_landmarks
                )
            )


        # =================================================
        # RESULT
        # =================================================

        return {
            "posture_result":
                self.latest_posture_result,

            "gesture_result":
                self.latest_gesture_result,

            "gaze_result":
                self.latest_gaze_result,

            "pose_landmarks":
                self.smoothed_landmarks,

            "face_landmarks":
                self.latest_face_landmarks,

            "pose_updated":
                pose_updated,

            "face_updated":
                face_updated,
        }


    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        self.pose_detector.close()
        self.face_detector.close()