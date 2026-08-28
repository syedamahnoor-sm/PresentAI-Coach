import time
from collections import deque
from statistics import median


class SessionManager:
    """
    Aggregates PresentAI Coach analysis results across
    an entire presentation session.

    Inputs:
        - posture results
        - gesture results
        - gaze results
        - final speech results

    Produces:
        - session-level dimension scores
        - overall presentation score
        - strengths
        - improvement areas
        - recommendations
        - session statistics

    Visual results are sampled at controlled intervals
    instead of storing every camera frame.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        sample_interval=1.0,
        recent_window_size=5,
    ):
        """
        Args:
            sample_interval:
                Minimum seconds between stored visual
                samples.

            recent_window_size:
                Number of recent samples used for
                short-term median stabilization.
        """

        self.sample_interval = sample_interval
        self.recent_window_size = recent_window_size

        # -------------------------------------------------
        # Overall score weights
        # -------------------------------------------------

        self.weights = {
            "posture": 0.25,
            "gesture": 0.20,
            "gaze": 0.25,
            "speech": 0.30,
        }

        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Start a fresh presentation session.
        """

        self.session_start_time = None
        self.session_end_time = None

        self.last_sample_time = None

        self.posture_scores = []
        self.gesture_scores = []
        self.gaze_scores = []

        self.recent_posture = deque(
            maxlen=self.recent_window_size
        )

        self.recent_gesture = deque(
            maxlen=self.recent_window_size
        )

        self.recent_gaze = deque(
            maxlen=self.recent_window_size
        )

        self.speech_result = None

        self.total_visual_updates = 0
        self.stored_visual_samples = 0


    # =====================================================
    # START SESSION
    # =====================================================

    def start_session(self):
        """
        Start a new presentation session.
        """

        self.reset()

        self.session_start_time = (
            time.monotonic()
        )


    # =====================================================
    # HELPER: VALID SCORE
    # =====================================================

    @staticmethod
    def _get_valid_score(
        result
    ):
        """
        Safely extract a score from an analyzer result.

        Returns None when the result is missing,
        invalid, or outside the expected 0-100 range.
        """

        if result is None:
            return None

        if not isinstance(
            result,
            dict
        ):
            return None

        score = result.get(
            "score"
        )

        if score is None:
            return None

        try:

            score = float(
                score
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if not (
            0.0
            <= score
            <= 100.0
        ):
            return None

        return score


    # =====================================================
    # UPDATE VISUAL RESULTS
    # =====================================================

    def update_visual(
        self,
        posture_result=None,
        gesture_result=None,
        gaze_result=None,
        timestamp=None,
    ):
        """
        Receive current visual analyzer results.

        This may be called every camera frame, but samples
        are stored only once per sample_interval.

        Returns:
            True  -> sample stored
            False -> skipped
        """

        if self.session_start_time is None:

            self.start_session()

        self.total_visual_updates += 1

        if timestamp is None:

            timestamp = (
                time.monotonic()
            )

        # -------------------------------------------------
        # Sampling interval
        # -------------------------------------------------

        if self.last_sample_time is not None:

            elapsed = (
                timestamp
                - self.last_sample_time
            )

            if (
                elapsed
                < self.sample_interval
            ):

                return False

        # -------------------------------------------------
        # Extract scores
        # -------------------------------------------------

        posture_score = (
            self._get_valid_score(
                posture_result
            )
        )

        gesture_score = (
            self._get_valid_score(
                gesture_result
            )
        )

        gaze_score = (
            self._get_valid_score(
                gaze_result
            )
        )

        # Don't count a sample if nothing valid exists.
        if (
            posture_score is None
            and gesture_score is None
            and gaze_score is None
        ):

            return False

        # -------------------------------------------------
        # Store valid scores independently
        # -------------------------------------------------

        if posture_score is not None:

            self.posture_scores.append(
                posture_score
            )

            self.recent_posture.append(
                posture_score
            )

        if gesture_score is not None:

            self.gesture_scores.append(
                gesture_score
            )

            self.recent_gesture.append(
                gesture_score
            )

        if gaze_score is not None:

            self.gaze_scores.append(
                gaze_score
            )

            self.recent_gaze.append(
                gaze_score
            )

        self.last_sample_time = timestamp

        self.stored_visual_samples += 1

        return True


    # =====================================================
    # ADD SPEECH RESULT
    # =====================================================

    def set_speech_result(
        self,
        speech_result
    ):
        """
        Store final Faster-Whisper speech analysis.
        """

        if not isinstance(
            speech_result,
            dict
        ):

            self.speech_result = None
            return False

        if not speech_result.get(
            "has_enough_data",
            False
        ):

            self.speech_result = (
                speech_result
            )

            return False

        score = self._get_valid_score(
            speech_result
        )

        if score is None:

            self.speech_result = None
            return False

        self.speech_result = (
            speech_result
        )

        return True


    # =====================================================
    # ROBUST SCORE AGGREGATION
    # =====================================================

    @staticmethod
    def _aggregate_scores(
        scores
    ):
        """
        Aggregate session scores robustly.

        Uses:
            70% mean
            30% median

        This keeps the result representative of the
        overall presentation while reducing the effect
        of occasional noisy frames.
        """

        if not scores:
            return None

        mean_score = (
            sum(scores)
            / len(scores)
        )

        median_score = median(
            scores
        )

        final_score = (
            0.70
            * mean_score

            +

            0.30
            * median_score
        )

        return round(
            max(
                0.0,
                min(
                    final_score,
                    100.0
                )
            ),
            1
        )


    # =====================================================
    # RECENT SCORES
    # =====================================================

    def get_recent_scores(self):
        """
        Return stabilized recent visual scores.

        Useful later for live UI display.
        """

        def recent_median(
            values
        ):

            if not values:
                return None

            return round(
                median(values),
                1
            )

        return {
            "posture":
                recent_median(
                    self.recent_posture
                ),

            "gesture":
                recent_median(
                    self.recent_gesture
                ),

            "gaze":
                recent_median(
                    self.recent_gaze
                ),
        }


    # =====================================================
    # STATUS
    # =====================================================

    @staticmethod
    def determine_status(
        score
    ):
        """
        Convert a score into a presentation status.
        """

        if score is None:
            return "Not Available"

        if score >= 85:
            return "Excellent"

        if score >= 70:
            return "Good"

        if score >= 50:
            return "Needs Improvement"

        return "Poor"


    # =====================================================
    # OVERALL SCORE
    # =====================================================

    def _calculate_overall_score(
        self,
        dimension_scores
    ):
        """
        Calculate weighted overall score.

        If a dimension is unavailable, its weight is
        redistributed across the available dimensions.
        """

        weighted_total = 0.0
        available_weight = 0.0

        for (
            dimension,
            weight
        ) in self.weights.items():

            score = dimension_scores.get(
                dimension
            )

            if score is None:
                continue

            weighted_total += (
                score
                * weight
            )

            available_weight += (
                weight
            )

        if available_weight == 0:
            return None

        overall_score = (
            weighted_total
            / available_weight
        )

        return round(
            overall_score,
            1
        )


    # =====================================================
    # STRENGTHS
    # =====================================================

    @staticmethod
    def _generate_strengths(
        dimension_scores
    ):
        """
        Identify strong presentation dimensions.
        """

        labels = {
            "posture":
                "Posture",

            "gesture":
                "Gestures",

            "gaze":
                "Eye Contact",

            "speech":
                "Speech Delivery",
        }

        strengths = []

        for (
            dimension,
            score
        ) in dimension_scores.items():

            if (
                score is not None
                and score >= 80
            ):

                strengths.append(
                    labels[
                        dimension
                    ]
                )

        # If nothing reaches the strong threshold,
        # return the strongest available dimension.
        if not strengths:

            available = {
                dimension: score
                for (
                    dimension,
                    score
                ) in dimension_scores.items()
                if score is not None
            }

            if available:

                strongest = max(
                    available,
                    key=available.get
                )

                strengths.append(
                    labels[
                        strongest
                    ]
                )

        return strengths


    # =====================================================
    # IMPROVEMENT AREAS
    # =====================================================

    @staticmethod
    def _generate_improvement_areas(
        dimension_scores
    ):
        """
        Identify dimensions that need attention.
        """

        labels = {
            "posture":
                "Posture",

            "gesture":
                "Gestures",

            "gaze":
                "Eye Contact",

            "speech":
                "Speech Delivery",
        }

        improvements = []

        for (
            dimension,
            score
        ) in dimension_scores.items():

            if (
                score is not None
                and score < 70
            ):

                improvements.append(
                    labels[
                        dimension
                    ]
                )

        return improvements


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    def _generate_recommendations(
        self,
        dimension_scores
    ):
        """
        Generate actionable final coaching advice.
        """

        recommendations = []

        posture = dimension_scores.get(
            "posture"
        )

        gesture = dimension_scores.get(
            "gesture"
        )

        gaze = dimension_scores.get(
            "gaze"
        )

        speech = dimension_scores.get(
            "speech"
        )

        # -------------------------------------------------
        # Posture
        # -------------------------------------------------

        if (
            posture is not None
            and posture < 70
        ):

            recommendations.append(
                "Maintain a more upright and stable "
                "posture throughout the presentation."
            )

        # -------------------------------------------------
        # Gestures
        # -------------------------------------------------

        if (
            gesture is not None
            and gesture < 70
        ):

            recommendations.append(
                "Use controlled, natural hand gestures "
                "to support your key points."
            )

        # -------------------------------------------------
        # Gaze
        # -------------------------------------------------

        if (
            gaze is not None
            and gaze < 70
        ):

            recommendations.append(
                "Look toward the camera more consistently "
                "to strengthen audience engagement."
            )

        # -------------------------------------------------
        # Speech
        # -------------------------------------------------

        if (
            speech is not None
            and speech < 70
        ):

            recommendations.append(
                "Improve speech delivery by maintaining "
                "a steady pace and reducing unnecessary "
                "fillers or long pauses."
            )

        # -------------------------------------------------
        # Use detailed speech feedback when available
        # -------------------------------------------------

        if self.speech_result is not None:

            speech_feedback = (
                self.speech_result.get(
                    "feedback",
                    []
                )
            )

            for item in speech_feedback:

                if (
                    item
                    and
                    item
                    != "Speech delivery sounds balanced."
                    and
                    item not in recommendations
                ):

                    recommendations.append(
                        item
                    )

        # -------------------------------------------------
        # Strong overall performance
        # -------------------------------------------------

        if not recommendations:

            recommendations.append(
                "Your presentation delivery was balanced. "
                "Keep practicing to maintain consistency."
            )

        # Keep final report focused.
        return recommendations[:3]


    # =====================================================
    # END SESSION
    # =====================================================

    def end_session(
        self
    ):
        """
        Finish session and generate the final report.
        """

        if self.session_start_time is None:

            return {
                "overall_score": None,
                "status": "No Session Data",
                "dimension_scores": {},
                "strengths": [],
                "improvement_areas": [],
                "recommendations": [],
            }

        self.session_end_time = (
            time.monotonic()
        )

        # -------------------------------------------------
        # Dimension scores
        # -------------------------------------------------

        posture_score = (
            self._aggregate_scores(
                self.posture_scores
            )
        )

        gesture_score = (
            self._aggregate_scores(
                self.gesture_scores
            )
        )

        gaze_score = (
            self._aggregate_scores(
                self.gaze_scores
            )
        )

        speech_score = None

        if self.speech_result is not None:

            if self.speech_result.get(
                "has_enough_data",
                False
            ):

                speech_score = (
                    self._get_valid_score(
                        self.speech_result
                    )
                )

                if speech_score is not None:

                    speech_score = round(
                        speech_score,
                        1
                    )

        dimension_scores = {
            "posture":
                posture_score,

            "gesture":
                gesture_score,

            "gaze":
                gaze_score,

            "speech":
                speech_score,
        }

        # -------------------------------------------------
        # Overall score
        # -------------------------------------------------

        overall_score = (
            self._calculate_overall_score(
                dimension_scores
            )
        )

        status = (
            self.determine_status(
                overall_score
            )
        )

        # -------------------------------------------------
        # Session duration
        # -------------------------------------------------

        duration_seconds = (
            self.session_end_time
            - self.session_start_time
        )

        # -------------------------------------------------
        # Report
        # -------------------------------------------------

        return {
            "overall_score":
                overall_score,

            "status":
                status,

            "dimension_scores":
                dimension_scores,

            "dimension_statuses": {
                dimension:
                    self.determine_status(
                        score
                    )

                for (
                    dimension,
                    score
                ) in dimension_scores.items()
            },

            "strengths":
                self._generate_strengths(
                    dimension_scores
                ),

            "improvement_areas":
                self._generate_improvement_areas(
                    dimension_scores
                ),

            "recommendations":
                self._generate_recommendations(
                    dimension_scores
                ),

            "session_duration_seconds":
                round(
                    duration_seconds,
                    1
                ),

            "visual_samples": {
                "posture":
                    len(
                        self.posture_scores
                    ),

                "gesture":
                    len(
                        self.gesture_scores
                    ),

                "gaze":
                    len(
                        self.gaze_scores
                    ),

                "stored_updates":
                    self.stored_visual_samples,

                "total_updates":
                    self.total_visual_updates,
            },

            "speech_metrics":
                self.speech_result,
        }