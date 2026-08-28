from speech_analyzer import SpeechAnalyzer
import time

def main():

    print()
    print(
        "===================================="
    )

    print(
        "PresentAI Speech Analyzer Test"
    )

    print(
        "===================================="
    )

    print()

    print(
        "Loading speech model..."
    )

    analyzer = SpeechAnalyzer(
        model_size="base.en",
        device="cpu",
        compute_type="int8",
    )

    print()
    print(
        "Model ready."
    )

    print()

    print(
        "Start speaking."
    )

    print(
        "Press Ctrl+C when finished."
    )

    print()

    try:

        analyzer.start()

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:

        print()

        print(
            "Stopping recording..."
        )

        analyzer.stop()

    print()

    # =====================================================
    # TRANSCRIBE
    # =====================================================

    transcript = (
        analyzer.transcribe()
    )

    # =====================================================
    # METRICS
    # =====================================================

    metrics = (
        analyzer.get_metrics()
    )

    # =====================================================
    # RESULTS
    # =====================================================

    print()

    print(
        "===================================="
    )

    print(
        "FINAL SPEECH RESULTS"
    )

    print(
        "===================================="
    )

    print(
        f"Speech Score: "
        f"{metrics['score']}/100"
    )

    print(
        f"Status: "
        f"{metrics['status']}"
    )

    print(
        f"WPM: "
        f"{metrics['wpm']}"
    )

    print(
        f"Words: "
        f"{metrics['word_count']}"
    )

    print(
        f"Fillers: "
        f"{metrics['filler_count']}"
    )

    print(
        f"Filler Rate: "
        f"{metrics['filler_rate']}%"
    )

    print(
        f"Pauses: "
        f"{metrics['pause_count']}"
    )

    print(
        f"Long Pauses: "
        f"{metrics['long_pause_count']}"
    )

    print(
        f"Fluency Score: "
        f"{metrics['fluency_score']}/100"
    )

    print(
        f"Recognition Confidence: "
        f"{metrics['recognition_confidence']}"
    )

    # =====================================================
    # FILLER BREAKDOWN
    # =====================================================

    if metrics[
        "filler_breakdown"
    ]:

        print()

        print(
            "Filler Breakdown:"
        )

        for (
            filler,
            count
        ) in metrics[
            "filler_breakdown"
        ].items():

            print(
                f"- {filler}: {count}"
            )

    # =====================================================
    # FEEDBACK
    # =====================================================

    print()

    print(
        "Feedback:"
    )

    for item in metrics[
        "feedback"
    ]:

        print(
            f"- {item}"
        )

    # =====================================================
    # TRANSCRIPT
    # =====================================================

    print()

    print(
        "Transcript:"
    )

    if transcript:

        print(
            transcript
        )

    else:

        print(
            "No speech detected."
        )


if __name__ == "__main__":
    main()