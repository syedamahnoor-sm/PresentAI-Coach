from ai_coach_agent import AICoachAgent


report = {

    "overall_score": 69.9,

    "status": "Needs Improvement",

    "dimension_scores": {

        "posture": 42.8,

        "gesture": 73.0,

        "gaze": 84.9,

        "speech": 78.0,
    },

    "speech_metrics": {

        "word_count": 53,

        "wpm": 214.0,

        "filler_count": 1,

        "filler_rate": 1.89,

        "pause_count": 0,

        "long_pause_count": 0,

        "fluency_score": 100.0,

        "recognition_confidence": 0.91,

        "has_enough_data": True,

        "transcript": (
            "You know a rocket ship when it takes off "
            "and flies into space, it uses 80 percent "
            "of its fuel. That's the same as a presentation. "
            "It takes so much energy and focus in the first "
            "three to four minutes that if you focus and "
            "get that right, the rest is much easier."
        ),
    }
}


print()
print(
    "=========================================="
)

print(
    "STARTING PRESENTAI AGENTIC AI COACH"
)

print(
    "=========================================="
)

print()


coach = AICoachAgent()


result = coach.analyze(
    report
)


print()
print(
    "=========================================="
)

print(
    "PRESENTAI AGENTIC COACHING RESULT"
)

print(
    "=========================================="
)

print()


print(
    "HEADLINE:"
)

print(
    result.get(
        "headline",
        "N/A"
    )
)


print()
print(
    "SUMMARY:"
)

print(
    result.get(
        "summary",
        "N/A"
    )
)


print()
print(
    "STRENGTHS:"
)

strengths = result.get(
    "strengths",
    []
)

if strengths:

    for strength in strengths:

        print(
            f"- {strength}"
        )

else:

    print(
        "- None returned"
    )


print()
print(
    "PRIORITY COACHING:"
)

priorities = result.get(
    "priority_coaching",
    []
)


if priorities:

    for index, item in enumerate(
        priorities,
        start=1
    ):

        print()

        print(
            f"{index}. "
            f"{item.get('area', 'Unknown')}"
        )

        print(
            "   Why:",
            item.get(
                "why_it_matters",
                ""
            )
        )

        print(
            "   How:",
            item.get(
                "how_to_improve",
                ""
            )
        )

else:

    print(
        "- No major priority returned"
    )


print()
print(
    "CONTENT FEEDBACK:"
)

content_feedback = result.get(
    "content_feedback"
)

if content_feedback:

    print(
        content_feedback
    )

else:

    print(
        "Not available."
    )


print()
print(
    "NEXT ATTEMPT PLAN:"
)

plan = result.get(
    "next_attempt_plan",
    []
)


if plan:

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}. {step}"
        )

else:

    print(
        "No plan returned."
    )


print()
print(
    "MAIN FOCUS:"
)

print(
    result.get(
        "focus_message",
        "N/A"
    )
)


print()
print(
    "=========================================="
)