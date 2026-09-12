import os
import tempfile
import threading
import time
import wave

import av
import cv2
import numpy as np

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

        # -------------------------------------------------
        # VISUAL ANALYSIS
        # -------------------------------------------------

        self.visual_analyzer = VisualAnalyzer(
            pose_every_n_frames=2,
            face_every_n_frames=4,
        )

        # -------------------------------------------------
        # SESSION MANAGER
        # -------------------------------------------------

        self.session_manager = SessionManager(
            sample_interval=1.0
        )

        # -------------------------------------------------
        # SPEECH ANALYZER
        # -------------------------------------------------
        #
        # IMPORTANT:
        #
        # SpeechAnalyzer is still the shared speech-analysis
        # engine used by both uploaded videos and live mode.
        #
        # We DO NOT call SpeechAnalyzer.start() here because
        # that uses sounddevice / PortAudio and therefore
        # attempts to open a microphone on the machine
        # running Python.
        #
        # On Streamlit Cloud that machine is the cloud
        # server, not the user's laptop.
        #
        # Live browser audio will instead arrive through
        # WebRTC and will eventually be passed into:
        #
        #     SpeechAnalyzer.transcribe_file()
        #
        # -------------------------------------------------

        self.speech_analyzer = SpeechAnalyzer(
            model_size="base.en",
            device="cpu",
            compute_type="int8",
        )

        # -------------------------------------------------
        # THREAD SAFETY
        # -------------------------------------------------

        # Protect visual/session state.
        self.lock = threading.Lock()

        # Protect browser audio state separately so that
        # incoming audio does not unnecessarily block
        # video-frame processing.
        self.audio_lock = threading.Lock()

        # -------------------------------------------------
        # SESSION STATE
        # -------------------------------------------------

        self.session_started = False
        self.session_finished = False

        # -------------------------------------------------
        # LIVE AUDIO STATE
        # -------------------------------------------------

        # All browser microphone audio is normalized into:
        #
        #   mono
        #   signed 16-bit PCM
        #   16000 Hz
        #
        # before being stored here.
        self.audio_chunks = []

        self.audio_sample_rate = 16000

        # PyAV performs WebRTC audio conversion/resampling.
        #
        # Browser/WebRTC audio commonly arrives at 48 kHz,
        # but Faster-Whisper works cleanly with our existing
        # 16 kHz speech pipeline.
        self.audio_resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.audio_sample_rate,
        )

        # -------------------------------------------------
        # LATEST VISUAL RESULTS
        # -------------------------------------------------

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

            # Reset aggregate presentation state.
            self.session_manager.start_session()

            # Reset speech analysis from any previous run.
            #
            # We intentionally DO NOT call:
            #
            #     self.speech_analyzer.start()
            #
            # because live browser microphone audio arrives
            # through WebRTC instead.
            self.speech_analyzer.reset()

            self.latest_results = {
                "posture_result": None,
                "gesture_result": None,
                "gaze_result": None,
            }

            self.session_started = True
            self.session_finished = False

        # Reset live browser audio independently.
        with self.audio_lock:

            self.audio_chunks = []

            # Re-create the resampler for each new session
            # so that no internal timing/buffer state from
            # the previous session is reused.
            self.audio_resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.audio_sample_rate,
            )


    # =====================================================
    # PROCESS WEBRTC AUDIO FRAME
    # =====================================================

    def process_audio_frame(self, frame):

        """
        Receive microphone audio from streamlit-webrtc.

        The incoming browser audio may use a different
        sample rate, channel layout or PCM format.

        PyAV converts it into:

            mono
            signed 16-bit PCM
            16000 Hz

        The converted PCM data is buffered until the live
        presentation session is finished.
        """

        # Quickly check session state.
        with self.lock:

            if (
                not self.session_started
                or self.session_finished
            ):
                return frame

        try:

            # -------------------------------------------------
            # RESAMPLE WEBRTC AUDIO
            # -------------------------------------------------

            with self.audio_lock:

                resampled_frames = (
                    self.audio_resampler.resample(
                        frame
                    )
                )

                if not resampled_frames:
                    return frame

                # PyAV can return one or more audio frames
                # for a single input frame.
                for resampled_frame in resampled_frames:

                    audio_array = (
                        resampled_frame.to_ndarray()
                    )

                    # Expected shape for mono PyAV audio is
                    # usually:
                    #
                    #     (1, samples)
                    #
                    # Flattening gives us normal PCM samples.
                    audio_array = np.asarray(
                        audio_array,
                        dtype=np.int16,
                    ).reshape(-1)

                    if audio_array.size == 0:
                        continue

                    self.audio_chunks.append(
                        audio_array.copy()
                    )

        except Exception as exc:

            # Do not crash the entire WebRTC presentation
            # because of one malformed audio frame.
            print(
                f"Browser audio frame error: {exc}"
            )

        # streamlit-webrtc expects an AudioFrame back when
        # an audio callback is used.
        return frame


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

            if (
                not self.session_started
                or self.session_finished
            ):

                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24"
                )

            results = (
                self.visual_analyzer.process_frame(
                    rgb_frame,
                    timestamp_ms
                )
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
                "posture_result":
                    posture_result,

                "gesture_result":
                    gesture_result,

                "gaze_result":
                    gaze_result,
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

                color = (
                    0,
                    255,
                    0
                )

            elif status == "Good":

                color = (
                    0,
                    220,
                    0
                )

            elif status == "Needs Improvement":

                color = (
                    0,
                    200,
                    255
                )

            else:

                color = (
                    0,
                    0,
                    255
                )

            cv2.putText(
                image,
                (
                    f"Posture: "
                    f"{status} "
                    f"({score}/100)"
                ),
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
                    landmark[0]
                    * width
                )

                y = int(
                    landmark[1]
                    * height
                )

                cv2.circle(
                    image,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1,
                )


            for (
                start_index,
                end_index
            ) in POSE_CONNECTIONS:

                if (
                    start_index
                    >= len(pose_landmarks)

                    or

                    end_index
                    >= len(pose_landmarks)
                ):
                    continue

                start = (
                    pose_landmarks[
                        start_index
                    ]
                )

                end = (
                    pose_landmarks[
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
                    ),
                )

                end_point = (
                    int(
                        end[0]
                        * width
                    ),

                    int(
                        end[1]
                        * height
                    ),
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

                if (
                    index
                    >= len(
                        face_landmarks
                    )
                ):
                    continue

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
    # BUILD TEMPORARY LIVE AUDIO FILE
    # =====================================================

    def _create_live_audio_file(self):

        """
        Combine buffered WebRTC microphone audio into
        one temporary 16-kHz mono WAV file.

        Returns:
            Path to temporary WAV file, or None when no
            browser microphone audio was captured.
        """

        with self.audio_lock:

            if not self.audio_chunks:
                return None

            audio_chunks = [
                chunk.copy()
                for chunk
                in self.audio_chunks
            ]

        combined_audio = np.concatenate(
            audio_chunks
        )

        if combined_audio.size == 0:
            return None

        temp_file = (
            tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            )
        )

        temp_audio_path = (
            temp_file.name
        )

        temp_file.close()

        try:

            with wave.open(
                temp_audio_path,
                "wb"
            ) as wav_file:

                wav_file.setnchannels(
                    1
                )

                # int16 = 2 bytes/sample
                wav_file.setsampwidth(
                    2
                )

                wav_file.setframerate(
                    self.audio_sample_rate
                )

                wav_file.writeframes(
                    combined_audio.astype(
                        np.int16
                    ).tobytes()
                )

        except Exception:

            if os.path.exists(
                temp_audio_path
            ):
                try:
                    os.remove(
                        temp_audio_path
                    )
                except OSError:
                    pass

            raise

        return temp_audio_path


    # =====================================================
    # STOP + BUILD FINAL REPORT
    # =====================================================

    def finish_session(self):

        # -------------------------------------------------
        # FREEZE SESSION
        # -------------------------------------------------

        with self.lock:

            if (
                not self.session_started
                or self.session_finished
            ):
                return None

            self.session_finished = True


        # -------------------------------------------------
        # BUILD SPEECH RESULT
        # -------------------------------------------------

        speech_result = None
        temp_audio_path = None

        try:

            temp_audio_path = (
                self._create_live_audio_file()
            )

            if temp_audio_path is not None:

                print(
                    "Transcribing live browser microphone..."
                )

                # Reuse the SAME Faster-Whisper pipeline
                # already used by uploaded-video audio.
                self.speech_analyzer.transcribe_file(
                    temp_audio_path
                )

                speech_result = (
                    self.speech_analyzer.get_metrics()
                )

            else:

                print(
                    "No live browser microphone "
                    "audio was captured."
                )

        except Exception as exc:

            # Visual report should still survive even if
            # speech transcription encounters a problem.
            print(
                f"Live speech analysis failed: {exc}"
            )

            speech_result = None

        finally:

            # Temporary microphone recording should never
            # remain on disk after transcription.
            if (
                temp_audio_path
                and os.path.exists(
                    temp_audio_path
                )
            ):

                try:

                    os.remove(
                        temp_audio_path
                    )

                except OSError:

                    pass


        # -------------------------------------------------
        # BUILD FINAL PRESENTATION REPORT
        # -------------------------------------------------

        with self.lock:

            if speech_result is not None:

                self.session_manager.set_speech_result(
                    speech_result
                )

            final_report = (
                self.session_manager.end_session()
            )

            # Keep report compatible with the existing
            # Streamlit result renderer.
            final_report[
                "speech_metrics"
            ] = speech_result

            self.session_started = False


        # -------------------------------------------------
        # CLEAR AUDIO MEMORY
        # -------------------------------------------------

        with self.audio_lock:

            self.audio_chunks = []

            self.audio_resampler = (
                av.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=self.audio_sample_rate,
                )
            )


        return final_report


    # =====================================================
    # CLEANUP
    # =====================================================

    def close(self):

        # No sounddevice stream exists in Streamlit Live
        # mode anymore, so there is no microphone device
        # to stop here.

        with self.lock:

            self.session_started = False
            self.session_finished = True

        with self.audio_lock:

            self.audio_chunks = []

        try:

            self.visual_analyzer.close()

        except Exception:

            pass