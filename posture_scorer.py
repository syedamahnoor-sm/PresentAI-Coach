import warnings

warnings.filterwarnings("ignore")

from collections import deque

import joblib
import numpy as np
import pandas as pd


class PostureScorer:
    """
    Convert classifier output + posture geometry
    into a stable continuous presentation-posture score.
    """

    def __init__(
        self,
        model_path="models/posture_classifier.pkl",
        history_size=15,
        smoothing_alpha=0.20,
    ):
        # =================================================
        # LOAD TRAINED MODEL
        # =================================================

        package = joblib.load(model_path)

        self.model = package["model"]
        self.feature_names = package["features"]

        # =================================================
        # IDENTIFY GOOD CLASS
        # =================================================

        self.classes = list(self.model.classes_)

        self.good_index = self.classes.index("good")

        # =================================================
        # TEMPORAL SMOOTHING
        # =================================================

        self.history = deque(maxlen=history_size)

        self.smoothing_alpha = smoothing_alpha

        self.smoothed_score = None

        self.current_status = None

    # =====================================================
    # FEATURE PREPARATION
    # =====================================================

    def prepare_features(self, features):
        """
        Convert one feature dictionary into the
        exact structure expected by the classifier.
        """

        row = {}

        for feature_name in self.feature_names:

            value = features.get(feature_name, np.nan)

            row[feature_name] = value

        dataframe = pd.DataFrame([row], columns=self.feature_names)

        return dataframe

    # =====================================================
    # ML PROBABILITY
    # =====================================================

    def get_good_probability(self, features):
        """
        Return classifier probability for GOOD posture.
        """

        X = self.prepare_features(features)

        probabilities = self.model.predict_proba(X)[0]

        good_probability = probabilities[self.good_index]

        return float(good_probability)

    # =====================================================
    # SAFE FEATURE VALUE
    # =====================================================

    def safe_feature(self, features, name, default=0.0):
        """
        Safely obtain a numerical feature.

        Handles:
        - missing values
        - None
        - NaN
        """

        value = features.get(name, default)

        if value is None:
            return default

        try:
            value = float(value)

        except (TypeError, ValueError):
            return default

        if np.isnan(value):
            return default

        return value

    # =====================================================
    # GEOMETRIC POSTURE SEVERITY
    # =====================================================

    def calculate_feature_severity(self, features):
        """
        Estimate severity of posture deviation.

        Returns:
            0.0 = neutral / low deviation
            1.0 = strong posture deviation

        We evaluate several independent posture cues
        and focus on the strongest few rather than
        averaging everything equally.
        """

        severity_parts = []

        # -------------------------------------------------
        # 1. SIDEWAYS TORSO LEAN
        # -------------------------------------------------

        torso_side_angle = abs(self.safe_feature(features, "torso_side_angle"))

        torso_side_severity = np.clip(torso_side_angle / 25.0, 0.0, 1.0)

        severity_parts.append(torso_side_severity)

        # -------------------------------------------------
        # 2. FORWARD / BACKWARD TORSO ANGLE
        # -------------------------------------------------

        torso_depth_angle = abs(self.safe_feature(features, "torso_depth_angle"))

        torso_depth_severity = np.clip(torso_depth_angle / 30.0, 0.0, 1.0)

        severity_parts.append(torso_depth_severity)

        # -------------------------------------------------
        # 3. HEAD SIDE OFFSET
        # -------------------------------------------------

        head_horizontal_offset = abs(
            self.safe_feature(features, "head_horizontal_offset")
        )

        head_horizontal_severity = np.clip(head_horizontal_offset / 0.60, 0.0, 1.0)

        severity_parts.append(head_horizontal_severity)

        # -------------------------------------------------
        # 4. HEAD FORWARD / DEPTH OFFSET
        # -------------------------------------------------

        head_depth_offset = abs(self.safe_feature(features, "head_depth_offset"))

        head_depth_severity = np.clip(head_depth_offset / 0.90, 0.0, 1.0)

        severity_parts.append(head_depth_severity)

        # -------------------------------------------------
        # 5. SHOULDER TILT
        # -------------------------------------------------

        shoulder_tilt = abs(self.safe_feature(features, "shoulder_tilt"))

        shoulder_tilt_severity = np.clip(shoulder_tilt / 20.0, 0.0, 1.0)

        severity_parts.append(shoulder_tilt_severity)

        # -------------------------------------------------
        # 6. HEAD TILT
        # -------------------------------------------------

        pose_head_tilt = abs(self.safe_feature(features, "pose_head_tilt"))

        head_tilt_severity = np.clip(pose_head_tilt / 20.0, 0.0, 1.0)

        severity_parts.append(head_tilt_severity)

        # -------------------------------------------------
        # 7. TORSO SIDE DISPLACEMENT
        # -------------------------------------------------

        torso_side_offset = abs(self.safe_feature(features, "torso_side_offset"))

        torso_side_offset_severity = np.clip(torso_side_offset / 0.50, 0.0, 1.0)

        severity_parts.append(torso_side_offset_severity)

        # -------------------------------------------------
        # 8. SHOULDER DEPTH ASYMMETRY
        # -------------------------------------------------

        shoulder_depth_difference = abs(
            self.safe_feature(features, "shoulder_depth_difference")
        )

        shoulder_depth_severity = np.clip(shoulder_depth_difference / 0.60, 0.0, 1.0)

        severity_parts.append(shoulder_depth_severity)

        # -------------------------------------------------
        # USE STRONGEST THREE DEVIATIONS
        # -------------------------------------------------

        severity_parts = sorted(severity_parts, reverse=True)

        strongest = severity_parts[:3]

        if not strongest:
            return 0.0

        severity = float(np.mean(strongest))

        return float(np.clip(severity, 0.0, 1.0))

    # =====================================================
    # HYBRID SCORE
    # =====================================================

    def calculate_hybrid_score(self, good_probability, features):
        """
        Combine:

        70% ML classifier probability
        30% geometric posture quality

        Geometry acts as a severity refinement,
        while ML remains the main signal.
        """

        severity = self.calculate_feature_severity(features)

        # ---------------------------------------------
        # ML component
        # ---------------------------------------------

        probability_score = np.clip(good_probability, 0.0, 1.0) * 100.0

        # ---------------------------------------------
        # Geometry component
        #
        # severity = 0 -> geometry score 100
        # severity = 1 -> geometry score 0
        # ---------------------------------------------

        geometry_score = (1.0 - severity) * 100.0

        # ---------------------------------------------
        # HYBRID
        # ---------------------------------------------

        hybrid_score = 0.70 * probability_score + 0.30 * geometry_score

        hybrid_score = float(np.clip(hybrid_score, 0.0, 100.0))

        return (hybrid_score, severity, probability_score, geometry_score)

    # =====================================================
    # TEMPORAL SMOOTHING
    # =====================================================

    def smooth_score(self, raw_score):
        """
        Stabilize frame-to-frame score.

        Stage 1:
            Rolling median removes sudden outliers.

        Stage 2:
            Exponential moving average makes genuine
            posture transitions gradual.
        """

        self.history.append(raw_score)

        median_score = float(np.median(self.history))

        if self.smoothed_score is None:

            self.smoothed_score = median_score

        else:

            self.smoothed_score = (
                self.smoothing_alpha * median_score
                + (1.0 - self.smoothing_alpha) * self.smoothed_score
            )

        return float(self.smoothed_score)

    # =====================================================
    # STATUS
    # =====================================================

    def determine_status(self, score):
        """
        Convert score into presentation-friendly status.

        Hysteresis prevents status flickering near
        category boundaries.
        """

        hysteresis = 3.0

        # -------------------------------------------------
        # FIRST STATUS
        # -------------------------------------------------

        if self.current_status is None:
            
            if score >= 85:
                self.current_status = "Excellent"
            elif score >= 65:
                self.current_status = "Good"

            elif score >= 35:
                self.current_status = "Needs Improvement"

            else:
                self.current_status = "Poor"

            return self.current_status
        # -------------------------------------------------
        # EXISTING STATUS: EXCELLENT
        # -------------------------------------------------

        if self.current_status == "Excellent":

            if score < (85 - hysteresis):

                self.current_status = "Good"

        # -------------------------------------------------
        # EXISTING STATUS: GOOD
        # -------------------------------------------------

        elif self.current_status == "Good":

            if score >= (85 + hysteresis):

                self.current_status = "Excellent"

            elif score < (65 - hysteresis):

                self.current_status = "Needs Improvement"

        # -------------------------------------------------
        # EXISTING STATUS: NEEDS IMPROVEMENT
        # -------------------------------------------------

        elif self.current_status == "Needs Improvement":

            if score >= (65 + hysteresis):

                self.current_status = "Good"

            elif score < (35 - hysteresis):

                self.current_status = "Poor"

        # -------------------------------------------------
        # EXISTING STATUS: POOR
        # -------------------------------------------------

        elif self.current_status == "Poor":

            if score >= (35 + hysteresis):

                self.current_status = "Needs Improvement"

        return self.current_status

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self, features):
        """
        Process one frame.

        Returns:
            ML probability
            geometric severity
            raw hybrid score
            smoothed score
            status
        """

        # ---------------------------------------------
        # ML probability
        # ---------------------------------------------

        good_probability = self.get_good_probability(features)

        # ---------------------------------------------
        # Hybrid score
        # ---------------------------------------------

        raw_score = float(np.clip(good_probability * 100.0, 0.0, 100.0))

        # ---------------------------------------------
        # Temporal smoothing
        # ---------------------------------------------

        stable_score = self.smooth_score(raw_score)

        stable_score = round(stable_score, 1)

        # ---------------------------------------------
        # Human-readable status
        # ---------------------------------------------

        status = self.determine_status(stable_score)

        return {
            "good_probability": round(good_probability, 3),
            "raw_score": round(raw_score, 1),
            "score": stable_score,
            "status": status,
        }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset temporal state when a presentation
        session ends or a new person/session begins.
        """

        self.history.clear()

        self.smoothed_score = None

        self.current_status = None
