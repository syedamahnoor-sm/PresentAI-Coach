import cv2
import time

from visual_analyzer import VisualAnalyzer
from session_manager import SessionManager
from speech_analyzer import SpeechAnalyzer


# =========================================================
# POSE CONNECTIONS - DISPLAY ONLY
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
# CREATE SHARED VISUAL ANALYZER
# =========================================================

# Performance optimization:
#
# Pose:
#   every 2 webcam frames
#
# Face:
#   every 4 webcam frames

visual_analyzer = VisualAnalyzer(
    pose_every_n_frames=2,
    face_every_n_frames=4,
)


# =========================================================
# SESSION MANAGER
# =========================================================

session_manager = SessionManager(
    sample_interval=1.0
)


# =========================================================
# SPEECH ANALYZER
# =========================================================

speech_analyzer = SpeechAnalyzer(
    model_size="base.en",
    device="cpu",
    compute_type="int8",
)


# =========================================================
# OPEN WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    visual_analyzer.close()

    raise RuntimeError(
        "Could not open webcam."
    )


# =========================================================
# START PRESENTATION SESSION
# =========================================================

session_manager.start_session()

speech_analyzer.start()


print()
print(
    "Presentation session started."
)

print(
    "Speak normally while presenting."
)

print(
    "Press Q to finish."
)

print()


# =========================================================
# MAIN WEBCAM LOOP
# =========================================================

try:

    while cap.isOpened():

        success, frame = (
            cap.read()
        )


        if not success:

            print(
                "Could not access webcam."
            )

            break


        # =================================================
        # MIRROR WEBCAM
        # =================================================

        frame = cv2.flip(
            frame,
            1
        )


        # =================================================
        # RGB CONVERSION
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # TIMESTAMP
        # =================================================

        timestamp_ms = (
            time.monotonic_ns()
            // 1_000_000
        )


        # =================================================
        # SHARED VISUAL ANALYSIS
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

        pose_landmarks = (
            results[
                "pose_landmarks"
            ]
        )

        face_landmarks = (
            results[
                "face_landmarks"
            ]
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
        # FRAME SIZE
        # =================================================

        height, width, _ = (
            frame.shape
        )


        # =================================================
        # POSTURE DISPLAY
        # =================================================

        if posture_result is not None:

            display_score = (
                posture_result[
                    "score"
                ]
            )

            display_status = (
                posture_result[
                    "status"
                ]
            )


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
                2,
            )


            cv2.putText(
                frame,
                (
                    "ML Good Confidence: "
                    f"{posture_result['good_probability']:.2f}"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )


        # =================================================
        # GESTURE DISPLAY
        # =================================================

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


            feedback = (
                gesture_result.get(
                    "feedback"
                )
            )


            if feedback:

                if isinstance(
                    feedback,
                    list
                ):

                    feedback_text = (
                        feedback[0]
                    )

                else:

                    feedback_text = (
                        str(
                            feedback
                        )
                    )


                cv2.putText(
                    frame,
                    feedback_text,
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )


        # =================================================
        # GAZE DISPLAY
        # =================================================

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


            feedback = (
                gaze_result.get(
                    "feedback"
                )
            )


            if feedback:

                if isinstance(
                    feedback,
                    list
                ):

                    feedback_text = (
                        feedback[0]
                    )

                else:

                    feedback_text = (
                        str(
                            feedback
                        )
                    )


                cv2.putText(
                    frame,
                    feedback_text,
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )


        # =================================================
        # DRAW POSE
        # =================================================

        if pose_landmarks is not None:

            for landmark in (
                pose_landmarks
            ):

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
                    -1,
                )


            # ---------------------------------------------
            # Skeleton
            # ---------------------------------------------

            for (
                start_index,
                end_index
            ) in POSE_CONNECTIONS:

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
                    frame,
                    start_point,
                    end_point,
                    (255, 255, 255),
                    2,
                )


        # =================================================
        # DRAW IMPORTANT FACE POINTS
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
                    -1,
                )


        # =================================================
        # NO POSE
        # =================================================

        if pose_landmarks is None:

            cv2.putText(
                frame,
                "No pose detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )


        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            (
                "PresentAI Coach - "
                "Live Presentation"
            ),
            frame
        )


        # =================================================
        # QUIT
        # =================================================

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            break


# =========================================================
# CLEAN UP VISUAL RESOURCES
# =========================================================

finally:

    cap.release()

    cv2.destroyAllWindows()

    visual_analyzer.close()


# =========================================================
# STOP SPEECH
# =========================================================

speech_analyzer.stop()


print()
print(
    "Presentation finished."
)

print(
    "Analyzing speech..."
)


# =========================================================
# TRANSCRIBE MICROPHONE AUDIO
# =========================================================

speech_analyzer.transcribe()


speech_result = (
    speech_analyzer.get_metrics()
)


# =========================================================
# ADD SPEECH TO SESSION
# =========================================================

session_manager.set_speech_result(
    speech_result
)


# =========================================================
# FINAL REPORT
# =========================================================

final_report = (
    session_manager.end_session()
)


# =========================================================
# PRINT REPORT
# =========================================================

print()
print(
    "===================================="
)

print(
    "PRESENTAI FINAL PRESENTATION REPORT"
)

print(
    "===================================="
)

print()


# ---------------------------------------------------------
# OVERALL
# ---------------------------------------------------------

if (
    final_report[
        "overall_score"
    ]
    is None
):

    print(
        "Overall Score: "
        "Not Available"
    )

else:

    print(
        f"Overall Score: "
        f"{final_report['overall_score']}/100"
    )


print(
    f"Status: "
    f"{final_report['status']}"
)

print()


# ---------------------------------------------------------
# DIMENSIONS
# ---------------------------------------------------------

print(
    "Dimension Scores:"
)


dimension_labels = {
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
) in final_report[
    "dimension_scores"
].items():

    label = (
        dimension_labels.get(
            dimension,
            dimension
        )
    )


    if score is None:

        print(
            f"- {label}: "
            "Not Available"
        )

    else:

        print(
            f"- {label}: "
            f"{score}/100"
        )


# ---------------------------------------------------------
# STRENGTHS
# ---------------------------------------------------------

print()
print(
    "Strengths:"
)


if final_report[
    "strengths"
]:

    for strength in (
        final_report[
            "strengths"
        ]
    ):

        print(
            f"- {strength}"
        )

else:

    print(
        "- None identified"
    )


# ---------------------------------------------------------
# IMPROVEMENT AREAS
# ---------------------------------------------------------

print()
print(
    "Improvement Areas:"
)


if final_report[
    "improvement_areas"
]:

    for area in (
        final_report[
            "improvement_areas"
        ]
    ):

        print(
            f"- {area}"
        )

else:

    print(
        "- None"
    )


# ---------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------

print()
print(
    "Recommendations:"
)


for recommendation in (
    final_report[
        "recommendations"
    ]
):

    print(
        f"- {recommendation}"
    )


# ---------------------------------------------------------
# SPEECH
# ---------------------------------------------------------

print()
print(
    "Speech Analysis:"
)


print(
    f"- Words: "
    f"{speech_result['word_count']}"
)

print(
    f"- WPM: "
    f"{speech_result['wpm']}"
)

print(
    f"- Fillers: "
    f"{speech_result['filler_count']}"
)

print(
    f"- Filler Rate: "
    f"{speech_result['filler_rate']}%"
)

print(
    f"- Pauses: "
    f"{speech_result['pause_count']}"
)

print(
    f"- Long Pauses: "
    f"{speech_result['long_pause_count']}"
)

print(
    f"- Fluency: "
    f"{speech_result['fluency_score']}/100"
)

print(
    f"- Recognition Confidence: "
    f"{speech_result['recognition_confidence']}"
)


# ---------------------------------------------------------
# VISUAL SAMPLE INFORMATION
# ---------------------------------------------------------

print()
print(
    "Visual Samples:"
)

print(
    final_report[
        "visual_samples"
    ]
)


# ---------------------------------------------------------
# TRANSCRIPT
# ---------------------------------------------------------

print()
print(
    "Transcript:"
)


if speech_result[
    "transcript"
]:

    print(
        speech_result[
            "transcript"
        ]
    )

else:

    print(
        "No speech recognized."
    )


print()
print(
    "===================================="
)