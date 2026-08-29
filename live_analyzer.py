import threading
import time

import av
import cv2

from visual_analyzer import VisualAnalyzer
from session_manager import SessionManager
from speech_analyzer import SpeechAnalyzer


# =========================================================
# DISPLAY CONNECTIONS
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
# LIVE PRESENTATION ANALYZER
# =========================================================

class LivePresentationAnalyzer:

    def __init__(self):

        self.visual_analyzer = VisualAnalyzer(
            pose_every_n_frames=2,
            face_every_n_frames=4,
        )

        self.session_manager = SessionManager(
            sample_interval=1.0
        )

        self.speech_analyzer = SpeechAnalyzer(
            model_size="base.en",
            device="cpu",
            compute_type="int8",
        )

        self.lock = threading.Lock()

        self.session_started = False
        self.session_finished = False

        self.latest_results = {
            "posture_result": None,
            "gesture_result": None,
            "gaze_result": None,
        }


    # =====================================================
    # START SESSION
    # =====================================================

    def start_session(self):

        with self.lock:

            if self.session_started:
                return

            self.session_manager.start_session()

            self.speech_analyzer.start()

            self.session_started = True
            self.session_finished = False


    # =====================================================
    # PROCESS WEBRTC VIDEO FRAME
    # =====================================================

    def process_frame(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Browser webcam behaves like a normal webcam,
        # so mirror the display.
        image = cv2.flip(
            image,
            1
        )

        rgb_frame = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        timestamp_ms = (
            time.monotonic_ns()
            // 1_000_000
        )

        with self.lock:

            if not self.session_started:
                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24"
                )

            results = self.visual_analyzer.process_frame(
                rgb_frame,
                timestamp_ms
            )

            posture_result = results.get(
                "posture_result"
            )

            gesture_result = results.get(
                "gesture_result"
            )

            gaze_result = results.get(
                "gaze_result"
            )

            pose_landmarks = results.get(
                "pose_landmarks"
            )

            face_landmarks = results.get(
                "face_landmarks"
            )

            self.latest_results = {
                "posture_result": posture_result,
                "gesture_result": gesture_result,
                "gaze_result": gaze_result,
            }

            self.session_manager.update_visual(
                posture_result=posture_result,
                gesture_result=gesture_result,
                gaze_result=gaze_result,
            )

        # =================================================
        # DRAW RESULTS
        # =================================================

        height, width, _ = image.shape


        # -------------------------------------------------
        # POSTURE
        # -------------------------------------------------

        if posture_result is not None:

            score = posture_result.get(
                "score",
                0
            )

            status = posture_result.get(
                "status",
                "Unknown"
            )

            if status == "Excellent":
                color = (0, 255, 0)

            elif status == "Good":
                color = (0, 220, 0)

            elif status == "Needs Improvement":
                color = (0, 200, 255)

            else:
                color = (0, 0, 255)

            cv2.putText(
                image,
                f"Posture: {status} ({score}/100)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )


        # -------------------------------------------------
        # GESTURES
        # -------------------------------------------------

        if gesture_result is not None:

            cv2.putText(
                image,
                (
                    f"Gestures: "
                    f"{gesture_result.get('status', 'Unknown')} "
                    f"({gesture_result.get('score', 0)}/100)"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )


        # -------------------------------------------------
        # GAZE
        # -------------------------------------------------

        if gaze_result is not None:

            cv2.putText(
                image,
                (
                    f"Eye Contact: "
                    f"{gaze_result.get('status', 'Unknown')} "
                    f"({gaze_result.get('score', 0)}/100)"
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2,
            )


        # -------------------------------------------------
        # POSE LANDMARKS
        # -------------------------------------------------

        if pose_landmarks is not None:

            for landmark in pose_landmarks:

                x = int(
                    landmark[0] * width
                )

                y = int(
                    landmark[1] * height
                )

                cv2.circle(
                    image,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1,
                )


            for start_index, end_index in POSE_CONNECTIONS:

                if (
                    start_index >= len(pose_landmarks)
                    or end_index >= len(pose_landmarks)
                ):
                    continue

                start = pose_landmarks[
                    start_index
                ]

                end = pose_landmarks[
                    end_index
                ]

                start_point = (
                    int(start[0] * width),
                    int(start[1] * height),
                )

                end_point = (
                    int(end[0] * width),
                    int(end[1] * height),
                )

                cv2.line(
                    image,
                    start_point,
                    end_point,
                    (255, 255, 255),
                    2,
                )


        # -------------------------------------------------
        # IMPORTANT FACE POINTS
        # -------------------------------------------------

        if face_landmarks is not None:

            important_face_points = [
                1,
                10,
                152,
                33,
                263,
            ]

            for index in important_face_points:

                if index >= len(face_landmarks):
                    continue

                landmark = face_landmarks[
                    index
                ]

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    image,
                    (x, y),
                    3,
                    (0, 255, 255),
                    -1,
                )


        # -------------------------------------------------
        # NO POSE
        # -------------------------------------------------

        if pose_landmarks is None:

            cv2.putText(
                image,
                "No pose detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )


        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


    # =====================================================
    # GET LATEST LIVE METRICS
    # =====================================================

    def get_latest_results(self):

        with self.lock:

            return {
                "posture_result":
                    self.latest_results[
                        "posture_result"
                    ],

                "gesture_result":
                    self.latest_results[
                        "gesture_result"
                    ],

                "gaze_result":
                    self.latest_results[
                        "gaze_result"
                    ],
            }


    # =====================================================
    # STOP + BUILD FINAL REPORT
    # =====================================================

    def finish_session(self):

        with self.lock:

            if (
                not self.session_started
                or self.session_finished
            ):
                return None

            self.session_finished = True

        # Stop microphone outside the visual lock.
        self.speech_analyzer.stop()

        # Faster-Whisper transcription can take time.
        self.speech_analyzer.transcribe()

        speech_result = (
            self.speech_analyzer.get_metrics()
        )

        with self.lock:

            self.session_manager.set_speech_result(
                speech_result
            )

            final_report = (
                self.session_manager.end_session()
            )

            # Make the report compatible with our
            # Streamlit results renderer.
            final_report[
                "speech_metrics"
            ] = speech_result

            self.session_started = False

        return final_report


    # =====================================================
    # CLEANUP
    # =====================================================

    def close(self):

        try:

            if self.session_started:

                self.speech_analyzer.stop()

        except Exception:

            pass

        try:

            self.visual_analyzer.close()

        except Exception:

            pass