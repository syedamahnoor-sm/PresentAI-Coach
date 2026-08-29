import argparse
import os

import cv2

from visual_analyzer import VisualAnalyzer
from speech_analyzer import SpeechAnalyzer
from session_manager import SessionManager
from media_utils import extract_audio_from_video


# =========================================================
# VIDEO ANALYSIS
# =========================================================

def analyze_video(
    video_path,
    show_preview=True,
    target_analysis_fps=10.0,
):
    """
    Analyze a prerecorded presentation.

    Visual:
        shared VisualAnalyzer

    Speech:
        original video audio -> Faster-Whisper
    """

    if not os.path.exists(
        video_path
    ):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if not fps or fps <= 0:

        fps = 30.0


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    duration_seconds = (

        total_frames
        / fps

        if total_frames > 0

        else 0.0
    )


    # =====================================================
    # ANALYSIS FRAME RATE
    # =====================================================

    frame_step = max(
        1,
        round(
            fps
            / target_analysis_fps
        )
    )


    actual_analysis_fps = (
        fps / frame_step
    )


    print()
    print(
        "===================================="
    )

    print(
        "PresentAI Uploaded Video Analysis"
    )

    print(
        "===================================="
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"Analysis FPS: "
        f"{actual_analysis_fps:.2f}"
    )

    print(
        f"Duration: "
        f"{duration_seconds:.1f} sec"
    )

    print()


    # =====================================================
    # SHARED COMPONENTS
    # =====================================================

    visual_analyzer = VisualAnalyzer(
        pose_every_n_frames=1,
        face_every_n_frames=2,
    )


    session_manager = SessionManager(
        sample_interval=1.0
    )


    session_manager.start_session()


    # =====================================================
    # VIDEO LOOP
    # =====================================================

    frame_index = 0
    analyzed_frames = 0

    stopped_early = False


    try:

        while cap.isOpened():

            success, frame = (
                cap.read()
            )


            if not success:
                break


            frame_index += 1


            # =================================================
            # SKIP UNNECESSARY FRAMES
            # =================================================

            if (
                frame_index - 1
            ) % frame_step != 0:

                continue


            analyzed_frames += 1


            timestamp_seconds = (
                (
                    frame_index - 1
                )
                / fps
            )


            timestamp_ms = int(
                timestamp_seconds
                * 1000
            )


            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # SHARED VISUAL ANALYZER
            # =================================================

            results = (
                visual_analyzer.process_frame(
                    rgb_frame,
                    timestamp_ms
                )
            )


            posture_result = (
                results[
                    "posture_result"
                ]
            )

            gesture_result = (
                results[
                    "gesture_result"
                ]
            )

            gaze_result = (
                results[
                    "gaze_result"
                ]
            )


            # =================================================
            # SESSION MANAGER
            # =================================================

            session_manager.update_visual(
                posture_result=posture_result,
                gesture_result=gesture_result,
                gaze_result=gaze_result,
                timestamp=timestamp_seconds,
            )


            # =================================================
            # PREVIEW
            # =================================================

            if show_preview:

                progress = 0.0


                if total_frames > 0:

                    progress = (
                        frame_index
                        / total_frames
                        * 100.0
                    )


                cv2.putText(
                    frame,
                    (
                        f"Analysis: "
                        f"{progress:.1f}%"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )


                if posture_result:

                    cv2.putText(
                        frame,
                        (
                            "Posture: "
                            f"{posture_result['score']}"
                        ),
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )


                if gesture_result:

                    cv2.putText(
                        frame,
                        (
                            "Gestures: "
                            f"{gesture_result['score']}"
                        ),
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )


                if gaze_result:

                    cv2.putText(
                        frame,
                        (
                            "Eye Contact: "
                            f"{gaze_result['score']}"
                        ),
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )


                cv2.imshow(
                    (
                        "PresentAI Coach - "
                        "Uploaded Video"
                    ),
                    frame
                )


                if (
                    cv2.waitKey(1)
                    & 0xFF
                    == ord("q")
                ):

                    stopped_early = True
                    break


    finally:

        cap.release()

        cv2.destroyAllWindows()

        visual_analyzer.close()


    # =====================================================
    # SPEECH
    # =====================================================

    speech_result = None

    audio_path = None


    if not stopped_early:

        print()
        print(
            "Extracting video audio..."
        )


        try:

            audio_path = (
                extract_audio_from_video(
                    video_path
                )
            )


            speech_analyzer = (
                SpeechAnalyzer(
                    model_size="base.en",
                    device="cpu",
                    compute_type="int8",
                )
            )


            speech_analyzer.transcribe_file(
                audio_path
            )


            speech_result = (
                speech_analyzer.get_metrics()
            )


            session_manager.set_speech_result(
                speech_result
            )


        except RuntimeError as error:

            print(
                f"Speech analysis skipped: "
                f"{error}"
            )


        finally:

            if (
                audio_path
                and
                os.path.exists(
                    audio_path
                )
            ):

                os.remove(
                    audio_path
                )


    # =====================================================
    # FINAL REPORT
    # =====================================================

    report = (
        session_manager.end_session()
    )


    report[
        "speech_metrics"
    ] = speech_result


    report[
        "video_metadata"
    ] = {
        "fps":
            round(
                fps,
                2
            ),

        "analysis_fps":
            round(
                actual_analysis_fps,
                2
            ),

        "total_frames":
            total_frames,

        "analyzed_frames":
            analyzed_frames,

        "duration_seconds":
            round(
                duration_seconds,
                1
            ),

        "stopped_early":
            stopped_early,
    }


    return report


# =========================================================
# REPORT
# =========================================================

def print_report(
    report
):

    print()
    print(
        "===================================="
    )

    print(
        "PRESENTAI PRESENTATION REPORT"
    )

    print(
        "===================================="
    )

    print()


    print(
        f"Overall Score: "
        f"{report['overall_score']}"
    )


    print(
        f"Status: "
        f"{report['status']}"
    )


    print()

    print(
        "Dimension Scores:"
    )


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


    for (
        dimension,
        score
    ) in report[
        "dimension_scores"
    ].items():

        print(
            f"- "
            f"{labels[dimension]}: "
            f"{score}"
        )


    print()

    print(
        "Recommendations:"
    )


    for recommendation in (
        report[
            "recommendations"
        ]
    ):

        print(
            f"- {recommendation}"
        )


    speech = report.get(
        "speech_metrics"
    )


    if speech:

        print()
        print(
            "Speech:"
        )

        print(
            f"- Words: "
            f"{speech['word_count']}"
        )

        print(
            f"- WPM: "
            f"{speech['wpm']}"
        )

        print(
            f"- Fillers: "
            f"{speech['filler_count']}"
        )

        print(
            f"- Long Pauses: "
            f"{speech['long_pause_count']}"
        )

        print()

        print(
            "Transcript:"
        )

        print(
            speech[
                "transcript"
            ]
        )


    print()

    print(
        "Performance:"
    )

    metadata = report[
        "video_metadata"
    ]


    print(
        f"- Original FPS: "
        f"{metadata['fps']}"
    )

    print(
        f"- Analysis FPS: "
        f"{metadata['analysis_fps']}"
    )

    print(
        f"- Frames analyzed: "
        f"{metadata['analyzed_frames']}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "video"
    )


    parser.add_argument(
        "--no-preview",
        action="store_true"
    )


    args = parser.parse_args()


    report = analyze_video(
        args.video,
        show_preview=(
            not args.no_preview
        ),
    )


    print_report(
        report
    )


if __name__ == "__main__":

    main()