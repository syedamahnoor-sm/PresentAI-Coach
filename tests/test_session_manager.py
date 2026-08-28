import time

from session_manager import SessionManager


def main():

    print()
    print(
        "===================================="
    )
    print(
        "PresentAI Session Manager Test"
    )
    print(
        "===================================="
    )
    print()

    manager = SessionManager(
        sample_interval=0.1
    )

    manager.start_session()

    # =====================================================
    # SIMULATED VISUAL ANALYSIS
    # =====================================================

    simulated_samples = [
        (88, 75, 82),
        (90, 78, 85),
        (86, 80, 79),
        (92, 76, 88),
        (89, 82, 84),
        (87, 79, 81),
        (91, 77, 86),
        (85, 81, 83),
        (88, 80, 85),
        (90, 78, 87),
    ]

    for (
        posture,
        gesture,
        gaze
    ) in simulated_samples:

        manager.update_visual(
            posture_result={
                "score": posture
            },

            gesture_result={
                "score": gesture
            },

            gaze_result={
                "score": gaze
            },
        )

        time.sleep(
            0.11
        )

    # =====================================================
    # SIMULATED SPEECH RESULT
    # =====================================================

    speech_result = {
        "score": 83.6,

        "status": "Good",

        "has_enough_data": True,

        "word_count": 35,

        "wpm": 111.3,

        "filler_count": 2,

        "filler_rate": 5.71,

        "pause_count": 1,

        "long_pause_count": 1,

        "feedback": [
            (
                "Reduce filler words such as "
                "um, uh, and like."
            )
        ],
    }

    manager.set_speech_result(
        speech_result
    )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    report = (
        manager.end_session()
    )

    print(
        f"Overall Score: "
        f"{report['overall_score']}/100"
    )

    print(
        f"Status: "
        f"{report['status']}"
    )

    print()

    print(
        "Dimension Scores:"
    )

    for (
        dimension,
        score
    ) in report[
        "dimension_scores"
    ].items():

        print(
            f"- {dimension}: {score}"
        )

    print()

    print(
        "Strengths:"
    )

    for strength in report[
        "strengths"
    ]:

        print(
            f"- {strength}"
        )

    print()

    print(
        "Improvement Areas:"
    )

    if report[
        "improvement_areas"
    ]:

        for area in report[
            "improvement_areas"
        ]:

            print(
                f"- {area}"
            )

    else:

        print(
            "- None"
        )

    print()

    print(
        "Recommendations:"
    )

    for recommendation in report[
        "recommendations"
    ]:

        print(
            f"- {recommendation}"
        )

    print()

    print(
        "Visual Samples:"
    )

    print(
        report[
            "visual_samples"
        ]
    )


if __name__ == "__main__":
    main()