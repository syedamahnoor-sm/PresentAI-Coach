"""Automated tests for PresentAI Coach's deterministic logic.

Run with:  pytest
No camera, microphone, model download or API key is needed.
"""

import pytest

from session_manager import SessionManager


# =========================================================
# SessionManager
# =========================================================

class TestSessionManager:

    @pytest.mark.parametrize(
        "score, expected",
        [
            (None, "Not Available"),
            (100, "Excellent"),
            (85, "Excellent"),
            (84.9, "Good"),
            (70, "Good"),
            (69.9, "Needs Improvement"),
            (50, "Needs Improvement"),
            (49.9, "Poor"),
            (0, "Poor"),
        ],
    )
    def test_status_thresholds(self, score, expected):
        assert SessionManager.determine_status(score) == expected

    def test_weights_sum_to_one(self):
        manager = SessionManager()
        assert sum(manager.weights.values()) == pytest.approx(1.0)

    def test_overall_score_uses_all_weights(self):
        manager = SessionManager()
        scores = {"posture": 80, "gesture": 70, "gaze": 90, "speech": 60}
        expected = 80 * 0.25 + 70 * 0.20 + 90 * 0.25 + 60 * 0.30
        assert manager._calculate_overall_score(scores) == pytest.approx(
            expected, abs=0.1
        )

    def test_missing_dimension_weight_is_redistributed(self):
        manager = SessionManager()
        scores = {"posture": 80, "gesture": None, "gaze": None, "speech": 60}
        expected = (80 * 0.25 + 60 * 0.30) / (0.25 + 0.30)
        assert manager._calculate_overall_score(scores) == pytest.approx(
            expected, abs=0.1
        )

    def test_no_scores_gives_none_not_zero(self):
        manager = SessionManager()
        empty = {"posture": None, "gesture": None, "gaze": None, "speech": None}
        assert manager._calculate_overall_score(empty) is None

    def test_empty_session_reports_not_available(self):
        manager = SessionManager(sample_interval=0.0)
        manager.start_session()
        report = manager.end_session()
        assert report["overall_score"] is None
        assert report["status"] == "Not Available"

    def test_full_session_report(self):
        manager = SessionManager(sample_interval=0.0)
        manager.start_session()

        for posture, gesture, gaze in [(88, 75, 82), (90, 78, 85),
                                       (86, 80, 79), (92, 76, 88)]:
            manager.update_visual(
                posture_result={"score": posture},
                gesture_result={"score": gesture},
                gaze_result={"score": gaze},
            )

        manager.set_speech_result({
            "score": 83.6, "status": "Good", "has_enough_data": True,
            "word_count": 35, "wpm": 111.3, "filler_count": 2,
            "filler_rate": 5.71, "pause_count": 1, "long_pause_count": 1,
            "feedback": ["Reduce filler words."],
        })

        report = manager.end_session()

        for key in ("overall_score", "status", "dimension_scores",
                    "strengths", "improvement_areas", "recommendations"):
            assert key in report

        dimensions = report["dimension_scores"]
        assert set(dimensions) == {"posture", "gesture", "gaze", "speech"}
        assert all(0 <= value <= 100 for value in dimensions.values())

        # The report must be consistent with its own parts
        assert report["overall_score"] == pytest.approx(
            manager._calculate_overall_score(dimensions), abs=0.1
        )
        assert report["status"] == SessionManager.determine_status(
            report["overall_score"]
        )


# =========================================================
# SpeechAnalyzer scoring (no microphone / Whisper needed)
# =========================================================

class TestSpeechScoring:

    @pytest.mark.parametrize(
        "wpm, expected",
        [
            (110, 100.0), (135, 100.0), (160, 100.0),   # ideal band
            (100, 85.0), (170, 85.0),
            (90, 65.0), (185, 65.0),
            (40, 45.0), (250, 45.0),                    # far too slow / fast
        ],
    )
    def test_pace_score(self, speech_analyzer, wpm, expected):
        assert speech_analyzer.calculate_pace_score(wpm) == expected

    @pytest.mark.parametrize(
        "fillers, words, expected",
        [
            (0, 0, 100.0),      # no speech: never divide by zero
            (0, 100, 100.0),
            (2, 100, 100.0),    # 2%  -> boundary
            (4, 100, 85.0),     # 4%
            (7, 100, 65.0),     # 7%
            (10, 100, 45.0),    # 10%
            (11, 100, 25.0),
        ],
    )
    def test_filler_score(self, speech_analyzer, fillers, words, expected):
        assert speech_analyzer.calculate_filler_score(fillers, words) == expected

    @pytest.mark.parametrize(
        "long_pauses, words, expected",
        [
            ([1, 2, 3], 5, 85.0),       # too little speech to judge
            ([], 50, 100.0),
            ([1, 2], 50, 85.0),
            ([1, 2, 3, 4], 50, 65.0),
            ([1] * 5, 50, 45.0),
        ],
    )
    def test_pause_score(self, speech_analyzer, long_pauses, words, expected):
        assert speech_analyzer.calculate_pause_score(long_pauses, words) == expected

    def test_fluency_is_weighted_blend(self, speech_analyzer):
        assert speech_analyzer.calculate_fluency_score(100, 50) == pytest.approx(80.0)

    @pytest.mark.parametrize(
        "score, expected",
        [(90, "Excellent"), (70, "Good"), (50, "Needs Improvement"), (10, "Poor")],
    )
    def test_status(self, speech_analyzer, score, expected):
        assert speech_analyzer.determine_status(score) == expected


class TestSpeechTranscriptAnalysis:

    def test_counts_single_and_phrase_fillers(self, speech_analyzer):
        speech_analyzer.words = "um so you know it is uh basically fine".split()
        total, breakdown = speech_analyzer.count_fillers()
        assert breakdown == {"um": 1, "uh": 1, "basically": 1, "you know": 1}
        assert total == 4

    def test_phrase_filler_needs_whole_words(self, speech_analyzer):
        # Regression: "i meant" used to be counted as the filler "i mean"
        speech_analyzer.words = "i meant to say that".split()
        total, _ = speech_analyzer.count_fillers()
        assert total == 0

    def test_no_words_no_fillers(self, speech_analyzer):
        speech_analyzer.words = []
        assert speech_analyzer.count_fillers() == (0, {})

    def test_wpm_ignores_initial_silence(self, speech_analyzer):
        # 60 words spoken between t=5s and t=35s -> 30s -> 120 WPM
        speech_analyzer.word_records = [
            {"start": 5 + i * 0.5, "end": 5.4 + i * 0.5} for i in range(60)
        ]
        wpm = speech_analyzer.calculate_wpm()
        assert wpm == pytest.approx(60 / ((5.4 + 59 * 0.5 - 5) / 60), rel=1e-6)
        assert 115 < wpm < 125

    def test_wpm_with_too_little_data_is_zero(self, speech_analyzer):
        speech_analyzer.word_records = [{"start": 0.0, "end": 0.4}]
        assert speech_analyzer.calculate_wpm() == 0.0


# =========================================================
# AI coach control flow (LLM replaced by a fake: no network)
# =========================================================

class _FakeResult:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _FakeResult(self.payload)


@pytest.fixture
def agent_class():
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_google_genai")
    from ai_coach_agent import AICoachAgent
    return AICoachAgent


class TestCoachAgentFlow:

    def _agent(self, agent_class, revised=None):
        agent = agent_class.__new__(agent_class)   # skip __init__: no API key / network
        agent.coaching_llm = _FakeLLM(revised or {"headline": "revised"})
        return agent

    def test_empty_report_never_calls_the_llm(self, agent_class):
        agent = self._agent(agent_class)
        result = agent.analyze({})
        assert result["headline"] == "No presentation data"

    def test_approved_draft_is_returned_unchanged(self, agent_class):
        agent = self._agent(agent_class)
        draft = {"headline": "Great job"}
        state = {"critique": {"approved": True}, "draft_coaching": draft, "evidence": {}}
        assert agent._finalize(state) == {"final_coaching": draft}
        assert agent.coaching_llm.calls == 0          # no needless second LLM call

    def test_rejected_draft_is_revised(self, agent_class):
        agent = self._agent(agent_class, revised={"headline": "Corrected"})
        state = {
            "critique": {"approved": False, "issues": ["invented a number"]},
            "draft_coaching": {"headline": "Draft"},
            "evidence": {},
        }
        result = agent._finalize(state)
        assert result == {"final_coaching": {"headline": "Corrected"}}
        assert agent.coaching_llm.calls == 1
