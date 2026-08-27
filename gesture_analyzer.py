from collections import deque
import math
import numpy as np


class GestureAnalyzer:
    """
    Analyze presentation gestures and movement over time.

    Produces:
        - gesture score 0-100
        - hand visibility
        - hand movement amount
        - body sway
        - frozen / low-movement detection
        - excessive movement detection
        - stable presentation-friendly status
    """

    def __init__(
        self,
        history_size=45,
        movement_window=15,
        score_smoothing_alpha=0.20,
    ):
        # -------------------------------------------------
        # Temporal history
        # -------------------------------------------------

        self.history_size = history_size
        self.movement_window = movement_window

        self.left_wrist_history = deque(
            maxlen=history_size
        )

        self.right_wrist_history = deque(
            maxlen=history_size
        )

        self.shoulder_center_history = deque(
            maxlen=history_size
        )

        self.hip_center_history = deque(
            maxlen=history_size
        )

        # -------------------------------------------------
        # Score smoothing
        # -------------------------------------------------

        self.score_smoothing_alpha = (
            score_smoothing_alpha
        )

        self.smoothed_score = None

        self.current_status = None


    # =====================================================
    # HELPERS
    # =====================================================

    def distance_2d(
        self,
        point_a,
        point_b
    ):
        return math.sqrt(
            (
                point_a[0]
                - point_b[0]
            ) ** 2
            +
            (
                point_a[1]
                - point_b[1]
            ) ** 2
        )


    def midpoint(
        self,
        point_a,
        point_b
    ):
        return [
            (
                point_a[0]
                + point_b[0]
            ) / 2,
            (
                point_a[1]
                + point_b[1]
            ) / 2,
            (
                point_a[2]
                + point_b[2]
            ) / 2,
        ]


    def safe_mean(
        self,
        values
    ):
        if not values:
            return 0.0

        return float(
            np.mean(values)
        )


    def calculate_path_movement(
        self,
        history
    ):
        """
        Calculate total movement over recent positions.
        """

        if len(history) < 2:
            return 0.0

        recent = list(
            history
        )[-self.movement_window:]

        movement = 0.0

        for i in range(
            1,
            len(recent)
        ):
            movement += (
                self.distance_2d(
                    recent[i - 1],
                    recent[i]
                )
            )

        return float(
            movement
        )


    def calculate_spread(
        self,
        history
    ):
        """
        Estimate how much a landmark moves around
        its recent average position.
        """

        if len(history) < 3:
            return 0.0

        recent = np.array(
            list(history)[
                -self.movement_window:
            ],
            dtype=float
        )

        center = np.mean(
            recent,
            axis=0
        )

        distances = np.sqrt(
            np.sum(
                (
                    recent[:, :2]
                    - center[:2]
                ) ** 2,
                axis=1
            )
        )

        return float(
            np.mean(distances)
        )


    # =====================================================
    # LANDMARK QUALITY
    # =====================================================

    def hand_visibility(
        self,
        pose_landmarks
    ):
        """
        Estimate whether each wrist is inside
        the visible frame.

        MediaPipe normalized coordinates:
            0 <= x <= 1
            0 <= y <= 1
        """

        left_wrist = pose_landmarks[15]
        right_wrist = pose_landmarks[16]

        left_visible = (
            0.0 <= left_wrist[0] <= 1.0
            and
            0.0 <= left_wrist[1] <= 1.0
        )

        right_visible = (
            0.0 <= right_wrist[0] <= 1.0
            and
            0.0 <= right_wrist[1] <= 1.0
        )

        visible_count = int(
            left_visible
        ) + int(
            right_visible
        )

        return (
            left_visible,
            right_visible,
            visible_count
        )


    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def update(
        self,
        pose_landmarks
    ):
        """
        Analyze one frame of pose landmarks.

        pose_landmarks:
            [[x, y, z], ...]
        """

        if (
            pose_landmarks is None
            or len(pose_landmarks) < 25
        ):
            return None


        # -------------------------------------------------
        # Important landmarks
        # -------------------------------------------------

        left_wrist = (
            pose_landmarks[15]
        )

        right_wrist = (
            pose_landmarks[16]
        )

        left_shoulder = (
            pose_landmarks[11]
        )

        right_shoulder = (
            pose_landmarks[12]
        )

        left_hip = (
            pose_landmarks[23]
        )

        right_hip = (
            pose_landmarks[24]
        )


        shoulder_center = (
            self.midpoint(
                left_shoulder,
                right_shoulder
            )
        )

        hip_center = (
            self.midpoint(
                left_hip,
                right_hip
            )
        )


        # -------------------------------------------------
        # Add to history
        # -------------------------------------------------

        self.left_wrist_history.append(
            left_wrist
        )

        self.right_wrist_history.append(
            right_wrist
        )

        self.shoulder_center_history.append(
            shoulder_center
        )

        self.hip_center_history.append(
            hip_center
        )


        # =================================================
        # 1. HAND VISIBILITY
        # =================================================

        (
            left_visible,
            right_visible,
            visible_hands
        ) = self.hand_visibility(
            pose_landmarks
        )


        # =================================================
        # 2. HAND MOVEMENT
        # =================================================

        left_hand_movement = (
            self.calculate_path_movement(
                self.left_wrist_history
            )
        )

        right_hand_movement = (
            self.calculate_path_movement(
                self.right_wrist_history
            )
        )


        total_hand_movement = (
            left_hand_movement
            + right_hand_movement
        )


        # =================================================
        # 3. BODY SWAY
        # =================================================

        shoulder_sway = (
            self.calculate_spread(
                self.shoulder_center_history
            )
        )

        hip_sway = (
            self.calculate_spread(
                self.hip_center_history
            )
        )


        body_sway = (
            shoulder_sway
            + hip_sway
        ) / 2.0


        # =================================================
        # 4. HAND POSITION / GESTURE ZONE
        # =================================================

        shoulder_width = (
            self.distance_2d(
                left_shoulder,
                right_shoulder
            )
        )


        shoulder_width = max(
            shoulder_width,
            1e-6
        )


        # Distance of hands from torso center
        left_hand_distance = (
            self.distance_2d(
                left_wrist,
                shoulder_center
            )
            / shoulder_width
        )

        right_hand_distance = (
            self.distance_2d(
                right_wrist,
                shoulder_center
            )
            / shoulder_width
        )


        average_hand_distance = (
            left_hand_distance
            + right_hand_distance
        ) / 2.0


        # =================================================
        # 5. MOVEMENT QUALITY FLAGS
        # =================================================

        enough_history = (
            len(
                self.left_wrist_history
            )
            >= self.movement_window
        )


        if enough_history:

            low_movement = (
                total_hand_movement
                < 0.18
            )

            excessive_hand_movement = (
                total_hand_movement
                > 2.00
            )

            excessive_body_sway = (
                body_sway
                > 0.035
            )

        else:

            low_movement = False

            excessive_hand_movement = False

            excessive_body_sway = False


        # =================================================
        # 6. GESTURE SCORE COMPONENTS
        # =================================================

        # -------------------------------------------------
        # Visibility score
        # -------------------------------------------------

        if visible_hands == 2:

            visibility_score = 100.0

        elif visible_hands == 1:

            visibility_score = 75.0

        else:

            visibility_score = 45.0


        # -------------------------------------------------
        # Movement score
        # -------------------------------------------------
        #
        # Moderate movement is ideal.
        # Very little movement looks frozen.
        # Too much movement looks distracting.
        # -------------------------------------------------

        if not enough_history:

            movement_score = 75.0

        elif total_hand_movement < 0.10:

            movement_score = 45.0

        elif total_hand_movement < 0.25:

            movement_score = 65.0

        elif total_hand_movement <= 1.20:

            movement_score = 100.0

        elif total_hand_movement <= 1.80:

            movement_score = 80.0

        else:

            movement_score = 55.0


        # -------------------------------------------------
        # Gesture-zone score
        # -------------------------------------------------
        #
        # Hands should not remain glued to torso,
        # but also should not stay extremely far away.
        # -------------------------------------------------

        if average_hand_distance < 0.45:

            gesture_zone_score = 65.0

        elif average_hand_distance <= 2.50:

            gesture_zone_score = 100.0

        elif average_hand_distance <= 3.50:

            gesture_zone_score = 80.0

        else:

            gesture_zone_score = 60.0


        # -------------------------------------------------
        # Stability score
        # -------------------------------------------------

        if body_sway < 0.012:

            stability_score = 100.0

        elif body_sway < 0.025:

            stability_score = 85.0

        elif body_sway < 0.040:

            stability_score = 65.0

        else:

            stability_score = 45.0


        # =================================================
        # 7. FINAL GESTURE SCORE
        # =================================================

        raw_score = (

            0.20
            * visibility_score

            +

            0.40
            * movement_score

            +

            0.20
            * gesture_zone_score

            +

            0.20
            * stability_score
        )


        raw_score = float(
            np.clip(
                raw_score,
                0.0,
                100.0
            )
        )


        # =================================================
        # 8. SMOOTH SCORE
        # =================================================

        if self.smoothed_score is None:

            self.smoothed_score = (
                raw_score
            )

        else:

            self.smoothed_score = (

                self.score_smoothing_alpha
                * raw_score

                +

                (
                    1.0
                    - self.score_smoothing_alpha
                )
                * self.smoothed_score
            )


        score = round(
            self.smoothed_score,
            1
        )


        # =================================================
        # 9. STATUS
        # =================================================

        status = self.determine_status(
            score
        )


        # =================================================
        # 10. FEEDBACK
        # =================================================

        feedback = []


        if enough_history:

            if low_movement:

                feedback.append(
                    "Use a few natural hand gestures."
                )


            if excessive_hand_movement:

                feedback.append(
                    "Reduce excessive hand movement."
                )


            if excessive_body_sway:

                feedback.append(
                    "Keep your body more stable."
                )


        if visible_hands == 0:

            feedback.append(
                "Keep your hands visible when possible."
            )


        if not feedback:

            feedback.append(
                "Gestures look balanced."
            )


        return {

            "score":
                score,

            "status":
                status,

            "visible_hands":
                visible_hands,

            "left_hand_movement":
                round(
                    left_hand_movement,
                    3
                ),

            "right_hand_movement":
                round(
                    right_hand_movement,
                    3
                ),

            "total_hand_movement":
                round(
                    total_hand_movement,
                    3
                ),

            "body_sway":
                round(
                    body_sway,
                    4
                ),

            "average_hand_distance":
                round(
                    average_hand_distance,
                    3
                ),

            "low_movement":
                low_movement,

            "excessive_hand_movement":
                excessive_hand_movement,

            "excessive_body_sway":
                excessive_body_sway,

            "feedback":
                feedback,
        }


    # =====================================================
    # STATUS WITH HYSTERESIS
    # =====================================================

    def determine_status(
        self,
        score
    ):
        """
        Stable gesture quality status.
        """

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


        if (
            self.current_status
            == "Excellent"
        ):

            if score < 82:

                self.current_status = (
                    "Good"
                )


        elif (
            self.current_status
            == "Good"
        ):

            if score >= 88:

                self.current_status = (
                    "Excellent"
                )

            elif score < 67:

                self.current_status = (
                    "Needs Improvement"
                )


        elif (
            self.current_status
            == "Needs Improvement"
        ):

            if score >= 73:

                self.current_status = (
                    "Good"
                )

            elif score < 47:

                self.current_status = (
                    "Poor"
                )


        elif (
            self.current_status
            == "Poor"
        ):

            if score >= 53:

                self.current_status = (
                    "Needs Improvement"
                )


        return self.current_status


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset gesture history for a new session/person.
        """

        self.left_wrist_history.clear()

        self.right_wrist_history.clear()

        self.shoulder_center_history.clear()

        self.hip_center_history.clear()

        self.smoothed_score = None

        self.current_status = None