from collections import deque
import numpy as np


class GazeAnalyzer:
    """
    Estimate presentation eye contact / gaze direction
    using MediaPipe Face Landmarker landmarks.

    This is a presentation heuristic, not medical-grade
    eye tracking.
    """

    def __init__(
        self,
        history_size=20,
        smoothing_alpha=0.20,
    ):
        self.history = deque(
            maxlen=history_size
        )

        self.smoothing_alpha = (
            smoothing_alpha
        )

        self.smoothed_score = None
        self.current_status = None


    # =====================================================
    # HELPERS
    # =====================================================

    def safe_value(
        self,
        value,
        default=0.0
    ):
        try:
            value = float(value)

            if np.isnan(value):
                return default

            return value

        except (
            TypeError,
            ValueError
        ):
            return default


    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def update(
        self,
        face_landmarks
    ):
        """
        Analyze one frame of face landmarks.

        Returns None when the face is unavailable.
        """

        if (
            face_landmarks is None
            or len(face_landmarks) < 455
        ):
            return None


        # -------------------------------------------------
        # Important MediaPipe face points
        # -------------------------------------------------

        nose = face_landmarks[1]

        left_eye_outer = face_landmarks[33]
        left_eye_inner = face_landmarks[133]

        right_eye_inner = face_landmarks[362]
        right_eye_outer = face_landmarks[263]

        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]

        forehead = face_landmarks[10]
        chin = face_landmarks[152]


        # =================================================
        # FACE CENTER
        # =================================================

        face_center_x = (
            left_cheek.x
            + right_cheek.x
        ) / 2.0


        # =================================================
        # 1. HORIZONTAL FACE TURN
        # =================================================

        face_width = max(
            abs(
                right_cheek.x
                - left_cheek.x
            ),
            1e-6
        )


        horizontal_offset = abs(
            nose.x
            - face_center_x
        ) / face_width


        # =================================================
        # 2. DEPTH ASYMMETRY
        # =================================================

        cheek_depth_difference = abs(
            left_cheek.z
            - right_cheek.z
        )


        # =================================================
        # 3. VERTICAL HEAD ORIENTATION
        # =================================================

        face_height = max(
            abs(
                chin.y
                - forehead.y
            ),
            1e-6
        )


        eye_center_y = (
            left_eye_outer.y
            + left_eye_inner.y
            + right_eye_inner.y
            + right_eye_outer.y
        ) / 4.0


        eye_vertical_ratio = abs(
            eye_center_y
            - forehead.y
        ) / face_height


        # =================================================
        # 4. SCORE COMPONENTS
        # =================================================

        # ---------------------------------------------
        # Horizontal score
        # ---------------------------------------------

        if horizontal_offset < 0.08:

            horizontal_score = 100.0

        elif horizontal_offset < 0.14:

            horizontal_score = 85.0

        elif horizontal_offset < 0.22:

            horizontal_score = 65.0

        else:

            horizontal_score = 40.0


        # ---------------------------------------------
        # Depth orientation score
        # ---------------------------------------------

        if cheek_depth_difference < 0.015:

            depth_score = 100.0

        elif cheek_depth_difference < 0.030:

            depth_score = 85.0

        elif cheek_depth_difference < 0.050:

            depth_score = 65.0

        else:

            depth_score = 40.0


        # ---------------------------------------------
        # Vertical score
        # ---------------------------------------------

        if 0.25 <= eye_vertical_ratio <= 0.48:

            vertical_score = 100.0

        elif 0.20 <= eye_vertical_ratio <= 0.55:

            vertical_score = 80.0

        else:

            vertical_score = 55.0


        # =================================================
        # 5. RAW GAZE SCORE
        # =================================================

        raw_score = (
            0.45 * horizontal_score
            + 0.35 * depth_score
            + 0.20 * vertical_score
        )


        # =================================================
        # 6. TEMPORAL SMOOTHING
        # =================================================

        self.history.append(
            raw_score
        )


        median_score = float(
            np.median(
                self.history
            )
        )


        if self.smoothed_score is None:

            self.smoothed_score = (
                median_score
            )

        else:

            self.smoothed_score = (
                self.smoothing_alpha
                * median_score

                + (
                    1.0
                    - self.smoothing_alpha
                )
                * self.smoothed_score
            )


        score = round(
            self.smoothed_score,
            1
        )


        # =================================================
        # 7. STATUS
        # =================================================

        status = self.determine_status(
            score
        )


        # =================================================
        # 8. FEEDBACK
        # =================================================

        if score >= 85:

            feedback = (
                "Strong eye contact."
            )

        elif score >= 70:

            feedback = (
                "Eye contact is generally good."
            )

        elif score >= 50:

            feedback = (
                "Look toward the camera more consistently."
            )

        else:

            feedback = (
                "Try to maintain more direct eye contact."
            )


        return {
            "score":
                score,

            "status":
                status,

            "horizontal_offset":
                round(
                    horizontal_offset,
                    3
                ),

            "cheek_depth_difference":
                round(
                    cheek_depth_difference,
                    4
                ),

            "eye_vertical_ratio":
                round(
                    eye_vertical_ratio,
                    3
                ),

            "feedback":
                feedback,
        }


    # =====================================================
    # STATUS
    # =====================================================

    def determine_status(
        self,
        score
    ):
        hysteresis = 3.0


        if self.current_status is None:

            if score >= 85:

                self.current_status = (
                    "Excellent"
                )

            elif score >= 70:

                self.current_status = (
                    "Good"
                )

            elif score >= 50:

                self.current_status = (
                    "Needs Improvement"
                )

            else:

                self.current_status = (
                    "Poor"
                )

            return self.current_status


        if self.current_status == "Excellent":

            if score < 82:

                self.current_status = "Good"


        elif self.current_status == "Good":

            if score >= 88:

                self.current_status = "Excellent"

            elif score < 67:

                self.current_status = (
                    "Needs Improvement"
                )


        elif self.current_status == "Needs Improvement":

            if score >= 73:

                self.current_status = "Good"

            elif score < 47:

                self.current_status = "Poor"


        elif self.current_status == "Poor":

            if score >= 53:

                self.current_status = (
                    "Needs Improvement"
                )


        return self.current_status


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        self.history.clear()

        self.smoothed_score = None
        self.current_status = None