import os
import tempfile
from collections import Counter

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


class SpeechAnalyzer:
    """
    PresentAI Coach speech analyzer using Faster-Whisper.

    Workflow:
        1. Record microphone audio
        2. Stop recording
        3. Transcribe with Faster-Whisper
        4. Extract word timestamps
        5. Analyze:
            - WPM
            - filler words
            - pauses
            - long pauses
            - fluency
            - final speech score
    """

    def __init__(
        self,
        model_size="base.en",
        sample_rate=16000,
        device="cpu",
        compute_type="int8",
        minimum_words=20,
    ):
        # =================================================
        # SETTINGS
        # =================================================

        self.sample_rate = sample_rate
        self.minimum_words = minimum_words

        # =================================================
        # WHISPER MODEL
        # =================================================

        print(
            f"Loading Faster-Whisper model: {model_size}"
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        print(
            "Faster-Whisper model loaded."
        )

        # =================================================
        # AUDIO RECORDING
        # =================================================

        self.audio_chunks = []

        self.stream = None
        self.recording = False

        # =================================================
        # TRANSCRIPTION DATA
        # =================================================

        self.words = []
        self.word_records = []
        self.transcript_segments = []

        # =================================================
        # FILLERS
        # =================================================

        self.single_fillers = {
            "um",
            "uh",
            "erm",
            "hmm",
            "like",
            "basically",
            "actually",
            "literally",
        }

        self.phrase_fillers = {
            "you know",
            "kind of",
            "sort of",
            "i mean",
        }


    # =====================================================
    # AUDIO CALLBACK
    # =====================================================

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):
        """
        Receive microphone audio continuously.
        """

        if status:
            print(
                f"Audio warning: {status}"
            )

        if self.recording:
            self.audio_chunks.append(
                indata.copy()
            )


    # =====================================================
    # START RECORDING
    # =====================================================

    def start(self):
        """
        Start microphone recording.
        """

        if self.recording:
            return

        self.reset()

        self.recording = True

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )

        self.stream.start()


    # =====================================================
    # STOP RECORDING
    # =====================================================

    def stop(self):
        """
        Stop microphone recording.
        """

        if not self.recording:
            return

        self.recording = False

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None


    # =====================================================
    # SAVE TEMPORARY AUDIO
    # =====================================================

    def _save_temp_audio(self):
        """
        Combine recorded chunks and save to temporary WAV.
        """

        if not self.audio_chunks:
            return None

        audio = np.concatenate(
            self.audio_chunks,
            axis=0
        )

        # Convert float32 [-1, 1] to int16.
        audio_int16 = np.int16(
            np.clip(
                audio,
                -1.0,
                1.0
            )
            * 32767
        )

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_path = temp_file.name

        temp_file.close()

        write(
            temp_path,
            self.sample_rate,
            audio_int16
        )

        return temp_path

    def _transcribe_audio_path(
        self,
        audio_path):
        self.words.clear()
        self.word_records.clear()
        self.transcript_segments.clear()

        if (
            not audio_path
            or
            not os.path.exists(
                audio_path
            )
        ):
            return ""

        print(
            "\nTranscribing speech..."
        )

        segments, info = (
            self.model.transcribe(
                audio_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )
        )


        for segment in segments:
            text = (
                segment.text.strip()
            )

            if text:

                self.transcript_segments.append(
                    text
                )


            if segment.words is None:
                continue


            for word_info in segment.words:

                word = (
                    word_info.word
                    .lower()
                    .strip()
                    .strip(
                        ".,!?;:\"'()[]{}"
                    )
                )

                if not word:
                    continue


                start = float(
                    word_info.start
                )

                end = float(
                    word_info.end
                )


                probability = getattr(
                    word_info,
                    "probability",
                    0.0
                )


                if probability is None:

                    probability = 0.0


                self.words.append(
                    word
                )


                self.word_records.append({
                    "word":
                        word,

                    "start":
                        start,

                    "end":
                        end,

                    "confidence":
                        float(
                            probability
                        ),
                })


        print(
            "Transcription complete."
            )


        return " ".join(
            self.transcript_segments
        )
    
    # =====================================================
    # TRANSCRIPTION
    # =====================================================

    def transcribe(self):
        """
            Transcribe microphone recording.
        """
        audio_path = (
            self._save_temp_audio()
    )

        if audio_path is None:
            return ""

        try:

            return (
                self._transcribe_audio_path(
                    audio_path
                )
            )

        finally:

            if (
                os.path.exists(
                    audio_path
                )
            ):

                os.remove(
                    audio_path
                )

    # =====================================================
    # TRANSCRIBE EXISTING AUDIO FILE
    # =====================================================

    def transcribe_file(
        self,
        audio_path
    ):
        """
        Transcribe an existing audio file.

        Used by uploaded presentation videos.
        """

        return (
            self._transcribe_audio_path(
                audio_path
            )
        )
    
    
    # =====================================================
    # FILLER ANALYSIS
    # =====================================================

    def count_fillers(self):
        """
        Count single and phrase filler words.
        """

        filler_counts = Counter()

        # ---------------------------------------------
        # Single-word fillers
        # ---------------------------------------------

        for word in self.words:

            if word in self.single_fillers:

                filler_counts[
                    word
                ] += 1

        # ---------------------------------------------
        # Phrase fillers
        # ---------------------------------------------

        transcript = " ".join(
            self.words
        )

        for phrase in self.phrase_fillers:

            count = transcript.count(
                phrase
            )

            if count > 0:

                filler_counts[
                    phrase
                ] += count

        total_fillers = sum(
            filler_counts.values()
        )

        return (
            total_fillers,
            dict(
                filler_counts
            )
        )


    # =====================================================
    # PAUSE ANALYSIS
    # =====================================================

    def analyze_pauses(self):
        """
        Detect pauses using gaps between word timestamps.

        Pause:
            >= 0.8 sec

        Long pause:
            >= 2.0 sec
        """

        pauses = []
        long_pauses = []

        if len(
            self.word_records
        ) < 2:

            return (
                pauses,
                long_pauses
            )

        for index in range(
            1,
            len(
                self.word_records
            )
        ):

            previous_word = (
                self.word_records[
                    index - 1
                ]
            )

            current_word = (
                self.word_records[
                    index
                ]
            )

            gap = (
                current_word["start"]
                - previous_word["end"]
            )

            if gap < 0:
                continue

            if gap >= 0.8:

                pauses.append(
                    gap
                )

            if gap >= 2.0:

                long_pauses.append(
                    gap
                )

        return (
            pauses,
            long_pauses
        )


    # =====================================================
    # SPEAKING PACE
    # =====================================================

    def calculate_wpm(self):
        """
        Calculate WPM using recognized speech span.

        Initial silence before the first word is ignored.
        """

        if len(
            self.word_records
        ) < 2:

            return 0.0

        first_word_start = (
            self.word_records[0]["start"]
        )

        last_word_end = (
            self.word_records[-1]["end"]
        )

        duration_seconds = (
            last_word_end
            - first_word_start
        )

        if duration_seconds <= 0:
            return 0.0

        duration_minutes = (
            duration_seconds
            / 60.0
        )

        return (
            len(
                self.word_records
            )
            / duration_minutes
        )


    # =====================================================
    # RECOGNITION CONFIDENCE
    # =====================================================

    def calculate_average_confidence(self):
        """
        Calculate average Whisper word probability.
        """

        if not self.word_records:
            return 0.0

        confidences = [
            record["confidence"]
            for record in self.word_records
            if record["confidence"] > 0
        ]

        if not confidences:
            return 0.0

        return (
            sum(confidences)
            / len(confidences)
        )


    # =====================================================
    # PACE SCORE
    # =====================================================

    def calculate_pace_score(
        self,
        wpm
    ):
        """
        Presentation-friendly speaking pace.

        Ideal:
            110-160 WPM
        """

        if 110 <= wpm <= 160:
            return 100.0

        if (
            95 <= wpm < 110
            or
            160 < wpm <= 175
        ):
            return 85.0

        if (
            80 <= wpm < 95
            or
            175 < wpm <= 190
        ):
            return 65.0

        return 45.0


    # =====================================================
    # FILLER SCORE
    # =====================================================

    def calculate_filler_score(
        self,
        filler_count,
        word_count
    ):
        """
        Score filler frequency.
        """

        if word_count == 0:
            return 100.0

        filler_rate = (
            filler_count
            / word_count
        ) * 100.0

        if filler_rate <= 2:
            return 100.0

        if filler_rate <= 4:
            return 85.0

        if filler_rate <= 7:
            return 65.0

        if filler_rate <= 10:
            return 45.0

        return 25.0


    # =====================================================
    # PAUSE SCORE
    # =====================================================

    def calculate_pause_score(
        self,
        long_pauses,
        word_count
    ):
        """
        Score long pause behavior.
        """

        if word_count < 10:
            return 85.0

        long_pause_count = len(
            long_pauses
        )

        if long_pause_count == 0:
            return 100.0

        if long_pause_count <= 2:
            return 85.0

        if long_pause_count <= 4:
            return 65.0

        return 45.0


    # =====================================================
    # FLUENCY SCORE
    # =====================================================

    def calculate_fluency_score(
        self,
        filler_score,
        pause_score
    ):
        """
        Simple presentation fluency estimate.
        """

        return (
            0.60
            * filler_score

            +

            0.40
            * pause_score
        )


    # =====================================================
    # STATUS
    # =====================================================

    def determine_status(
        self,
        score
    ):
        if score >= 85:
            return "Excellent"

        if score >= 70:
            return "Good"

        if score >= 50:
            return "Needs Improvement"

        return "Poor"


    # =====================================================
    # FEEDBACK
    # =====================================================

    def generate_feedback(
        self,
        wpm,
        filler_count,
        word_count,
        long_pauses,
        has_enough_data
    ):
        """
        Generate focused presentation feedback.
        """

        if not has_enough_data:

            remaining = max(
                0,
                self.minimum_words
                - word_count
            )

            return [
                (
                    f"Keep speaking - {remaining} "
                    "more recognized words needed "
                    "for scoring."
                )
            ]

        feedback = []

        # ---------------------------------------------
        # Pace
        # ---------------------------------------------

        if wpm < 100:

            feedback.append(
                "Try speaking slightly faster."
            )

        elif wpm > 170:

            feedback.append(
                "Slow your speaking pace slightly."
            )

        # ---------------------------------------------
        # Fillers
        # ---------------------------------------------

        if word_count > 0:

            filler_rate = (
                filler_count
                / word_count
            ) * 100.0

            if filler_rate > 5:

                feedback.append(
                    "Reduce filler words such as "
                    "um, uh, and like."
                )

        # ---------------------------------------------
        # Long pauses
        # ---------------------------------------------

        if len(
            long_pauses
        ) > 2:

            feedback.append(
                "Avoid too many long pauses."
            )

        # ---------------------------------------------
        # No major issues
        # ---------------------------------------------

        if not feedback:

            feedback.append(
                "Speech delivery sounds balanced."
            )

        return feedback


    # =====================================================
    # METRICS
    # =====================================================

    def get_metrics(self):
        """
        Calculate final speech metrics.

        Call transcribe() before this method.
        """

        word_count = len(
            self.words
        )

        has_enough_data = (
            word_count
            >= self.minimum_words
        )

        # ---------------------------------------------
        # WPM
        # ---------------------------------------------

        wpm = self.calculate_wpm()

        # ---------------------------------------------
        # Fillers
        # ---------------------------------------------

        (
            filler_count,
            filler_breakdown
        ) = self.count_fillers()

        # ---------------------------------------------
        # Pauses
        # ---------------------------------------------

        (
            pauses,
            long_pauses
        ) = self.analyze_pauses()

        # ---------------------------------------------
        # Component scores
        # ---------------------------------------------

        pace_score = (
            self.calculate_pace_score(
                wpm
            )
        )

        filler_score = (
            self.calculate_filler_score(
                filler_count,
                word_count
            )
        )

        pause_score = (
            self.calculate_pause_score(
                long_pauses,
                word_count
            )
        )

        fluency_score = (
            self.calculate_fluency_score(
                filler_score,
                pause_score
            )
        )

        # ---------------------------------------------
        # Final score
        # ---------------------------------------------

        if has_enough_data:

            speech_score = (

                0.40
                * pace_score

                +

                0.25
                * filler_score

                +

                0.15
                * pause_score

                +

                0.20
                * fluency_score
            )

            speech_score = round(
                max(
                    0.0,
                    min(
                        speech_score,
                        100.0
                    )
                ),
                1
            )

            status = self.determine_status(
                speech_score
            )

        else:

            speech_score = 0.0

            status = (
                "Insufficient Speech Data"
            )

        # ---------------------------------------------
        # Filler rate
        # ---------------------------------------------

        filler_rate = 0.0

        if word_count > 0:

            filler_rate = (
                filler_count
                / word_count
            ) * 100.0

        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        confidence = (
            self.calculate_average_confidence()
        )

        # ---------------------------------------------
        # Feedback
        # ---------------------------------------------

        feedback = self.generate_feedback(
            wpm,
            filler_count,
            word_count,
            long_pauses,
            has_enough_data
        )

        transcript = " ".join(
            self.transcript_segments
        )

        return {
            "score":
                speech_score,

            "status":
                status,

            "has_enough_data":
                has_enough_data,

            "word_count":
                word_count,

            "wpm":
                round(
                    wpm,
                    1
                ),

            "filler_count":
                filler_count,

            "filler_rate":
                round(
                    filler_rate,
                    2
                ),

            "filler_breakdown":
                filler_breakdown,

            "pause_count":
                len(
                    pauses
                ),

            "long_pause_count":
                len(
                    long_pauses
                ),

            "pace_score":
                round(
                    pace_score,
                    1
                ),

            "filler_score":
                round(
                    filler_score,
                    1
                ),

            "pause_score":
                round(
                    pause_score,
                    1
                ),

            "fluency_score":
                round(
                    fluency_score,
                    1
                ),

            "recognition_confidence":
                round(
                    confidence,
                    3
                ),

            "feedback":
                feedback,

            "transcript":
                transcript,
        }


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset analyzer for another presentation.
        """

        self.audio_chunks.clear()

        self.words.clear()

        self.word_records.clear()

        self.transcript_segments.clear()