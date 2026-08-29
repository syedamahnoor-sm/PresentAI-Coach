import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# STRUCTURED OUTPUT MODELS
# =========================================================

class DeliveryAnalysis(BaseModel):
    visual_summary: str
    strongest_visual_area: Optional[str] = None
    weakest_visual_area: Optional[str] = None
    observations: list[str] = Field(default_factory=list)
    recommended_focus: str


class SpeechAnalysis(BaseModel):
    speech_summary: str
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    recommended_focus: Optional[str] = None


class ContentAnalysis(BaseModel):
    available: bool
    summary: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class PriorityItem(BaseModel):
    rank: int
    area: str
    reason: str
    action: str


class PriorityAnalysis(BaseModel):
    priorities: list[PriorityItem] = Field(default_factory=list)
    preserve: list[str] = Field(default_factory=list)
    strategy_summary: str


class CoachingPriority(BaseModel):
    area: str
    why_it_matters: str
    how_to_improve: str


class CoachingResponse(BaseModel):
    headline: str
    summary: str

    strengths: list[str] = Field(
        default_factory=list
    )

    priority_coaching: list[CoachingPriority] = Field(
        default_factory=list
    )

    content_feedback: Optional[str] = None

    next_attempt_plan: list[str] = Field(
        default_factory=list
    )

    focus_message: str


class CritiqueResponse(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(
        default_factory=list
    )


# =========================================================
# LANGGRAPH STATE
# =========================================================

class CoachState(TypedDict, total=False):

    report: dict

    evidence: dict

    delivery_analysis: dict

    speech_analysis: dict

    content_analysis: dict

    priority_analysis: dict

    draft_coaching: dict

    critique: dict

    final_coaching: dict


# =========================================================
# PRESENTAI COACH AGENT
# =========================================================

class AICoachAgent:

    def __init__(
        self,
        model_name: Optional[str] = None
    ):

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY was not found.\n"
                "Add it to your .env file:\n"
                "GOOGLE_API_KEY=your_key_here"
            )

        self.model_name = (
            model_name
            or os.getenv(
                "PRESENTAI_LLM_MODEL",
                "gemini-3.6-flash"
            )
        )

        print(
            f"PresentAI Coach model: "
            f"{self.model_name}"
        )

        # Main Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.35,
            thinking_budget=512,
        )

        # Structured-output versions
        self.delivery_llm = (
            self.llm.with_structured_output(
                DeliveryAnalysis,
                method="json_schema"
            )
        )

        self.speech_llm = (
            self.llm.with_structured_output(
                SpeechAnalysis,
                method="json_schema"
            )
        )

        self.content_llm = (
            self.llm.with_structured_output(
                ContentAnalysis,
                method="json_schema"
            )
        )

        self.priority_llm = (
            self.llm.with_structured_output(
                PriorityAnalysis,
                method="json_schema"
            )
        )

        self.coaching_llm = (
            self.llm.with_structured_output(
                CoachingResponse,
                method="json_schema"
            )
        )

        self.critique_llm = (
            self.llm.with_structured_output(
                CritiqueResponse,
                method="json_schema"
            )
        )

        self.graph = self._build_graph()


    # =====================================================
    # PUBLIC METHOD
    # =====================================================

    def analyze(
        self,
        report: dict
    ) -> dict:

        if not report:
            return {
                "headline":
                    "No presentation data",

                "summary":
                    "Complete a presentation before "
                    "requesting AI coaching.",

                "strengths": [],

                "priority_coaching": [],

                "content_feedback": None,

                "next_attempt_plan": [],

                "focus_message":
                    "Complete a presentation first."
            }

        state: CoachState = {
            "report": report
        }

        result = self.graph.invoke(
            state
        )

        return result.get(
            "final_coaching",
            {}
        )


    # =====================================================
    # GRAPH DEFINITION
    # =====================================================

    def _build_graph(self):

        graph = StateGraph(
            CoachState
        )

        graph.add_node(
            "prepare_evidence",
            self._prepare_evidence
        )

        graph.add_node(
            "analyze_delivery",
            self._analyze_delivery
        )

        graph.add_node(
            "analyze_speech",
            self._analyze_speech
        )

        graph.add_node(
            "analyze_content",
            self._analyze_content
        )

        graph.add_node(
            "prioritize",
            self._prioritize
        )

        graph.add_node(
            "generate_coaching",
            self._generate_coaching
        )

        graph.add_node(
            "critique",
            self._critique
        )

        graph.add_node(
            "finalize",
            self._finalize
        )


        graph.add_edge(
            START,
            "prepare_evidence"
        )

        graph.add_edge(
            "prepare_evidence",
            "analyze_delivery"
        )

        graph.add_edge(
            "analyze_delivery",
            "analyze_speech"
        )

        graph.add_edge(
            "analyze_speech",
            "analyze_content"
        )

        graph.add_edge(
            "analyze_content",
            "prioritize"
        )

        graph.add_edge(
            "prioritize",
            "generate_coaching"
        )

        graph.add_edge(
            "generate_coaching",
            "critique"
        )

        graph.add_edge(
            "critique",
            "finalize"
        )

        graph.add_edge(
            "finalize",
            END
        )

        return graph.compile()


    # =====================================================
    # NODE 1
    # PREPARE OBJECTIVE EVIDENCE
    # =====================================================

    def _prepare_evidence(
        self,
        state: CoachState
    ):

        print(
            "Agent: preparing objective evidence..."
        )

        report = state.get(
            "report",
            {}
        )

        dimensions = (
            report.get(
                "dimension_scores",
                {}
            )
            or {}
        )

        speech = (
            report.get(
                "speech_metrics",
                {}
            )
            or {}
        )


        evidence = {

            "overall_score":
                report.get(
                    "overall_score"
                ),

            "overall_status":
                report.get(
                    "status"
                ),

            # VISUAL
            "posture_score":
                dimensions.get(
                    "posture"
                ),

            "gesture_score":
                dimensions.get(
                    "gesture"
                ),

            "gaze_score":
                dimensions.get(
                    "gaze"
                ),

            # SPEECH
            "speech_score":
                dimensions.get(
                    "speech"
                ),

            "word_count":
                speech.get(
                    "word_count",
                    0
                ),

            "wpm":
                speech.get(
                    "wpm"
                ),

            "filler_count":
                speech.get(
                    "filler_count"
                ),

            "filler_rate":
                speech.get(
                    "filler_rate"
                ),

            "pause_count":
                speech.get(
                    "pause_count"
                ),

            "long_pause_count":
                speech.get(
                    "long_pause_count"
                ),

            "fluency_score":
                speech.get(
                    "fluency_score"
                ),

            "recognition_confidence":
                speech.get(
                    "recognition_confidence"
                ),

            "has_enough_speech":
                speech.get(
                    "has_enough_data",
                    (
                        speech.get(
                            "word_count",
                            0
                        ) >= 20
                    )
                ),

            "transcript":
                speech.get(
                    "transcript",
                    ""
                )
        }


        return {
            "evidence": evidence
        }


    # =====================================================
    # NODE 2
    # VISUAL DELIVERY SPECIALIST
    # =====================================================

    def _analyze_delivery(
        self,
        state: CoachState
    ):

        print(
            "Agent: analyzing visual delivery..."
        )

        evidence = state[
            "evidence"
        ]

        visual_payload = {

            "posture_score":
                evidence.get(
                    "posture_score"
                ),

            "gesture_score":
                evidence.get(
                    "gesture_score"
                ),

            "gaze_score":
                evidence.get(
                    "gaze_score"
                )
        }


        prompt = f"""
You are the Visual Delivery Specialist
inside PresentAI Coach.

Analyze ONLY the objective presentation
analytics provided below.

OBJECTIVE DATA:
{visual_payload}

Your job is to reason about:

- presentation alignment/posture
- hand gesture effectiveness
- camera-facing gaze behavior
- relative strength between the dimensions
- which visual behavior matters most
  for the next attempt

IMPORTANT RULES:

1. Never invent scores.
2. Never alter the supplied scores.
3. Do not make medical, ergonomic,
   or health diagnoses.
4. "Posture" refers only to visible
   presentation alignment.
5. A strong score should normally be
   preserved, not criticized.
6. Do not claim something was observed
   unless the metrics support it.
7. Be concise and actionable.
8. If a metric is missing, do not infer it.

Return a structured analysis.
"""

        result = self.delivery_llm.invoke(
            prompt
        )


        return {

            "delivery_analysis":
                result.model_dump()
        }


    # =====================================================
    # NODE 3
    # SPEECH DELIVERY SPECIALIST
    # =====================================================

    def _analyze_speech(
        self,
        state: CoachState
    ):

        print(
            "Agent: analyzing speech delivery..."
        )

        evidence = state[
            "evidence"
        ]


        has_enough_speech = (
            evidence.get(
                "has_enough_speech",
                False
            )
        )


        if not has_enough_speech:

            return {

                "speech_analysis": {

                    "speech_summary":
                        "There was not enough recognized "
                        "speech for reliable speech coaching.",

                    "strengths":
                        [],

                    "concerns":
                        [],

                    "recommended_focus":
                        None
                }
            }


        speech_payload = {

            "speech_score":
                evidence.get(
                    "speech_score"
                ),

            "word_count":
                evidence.get(
                    "word_count"
                ),

            "wpm":
                evidence.get(
                    "wpm"
                ),

            "filler_count":
                evidence.get(
                    "filler_count"
                ),

            "filler_rate":
                evidence.get(
                    "filler_rate"
                ),

            "pause_count":
                evidence.get(
                    "pause_count"
                ),

            "long_pause_count":
                evidence.get(
                    "long_pause_count"
                ),

            "fluency_score":
                evidence.get(
                    "fluency_score"
                ),

            "recognition_confidence":
                evidence.get(
                    "recognition_confidence"
                )
        }


        prompt = f"""
You are the Speech Delivery Specialist
inside PresentAI Coach.

Evaluate ONLY this supplied speech evidence:

{speech_payload}

Reason about the COMBINATION of metrics,
not one isolated threshold.

Guidelines:

- Roughly 110-170 WPM is commonly
  comfortable for many presentations,
  although context matters.
- A very high WPM may reduce clarity.
- A low filler rate is a strength and
  should not be criticized.
- Pauses can be intentional and useful.
- Long pauses may matter when excessive.
- High fluency should be preserved.
- Recognition confidence should affect
  how strongly you phrase conclusions.

RULES:

1. Never invent metrics.
2. Never modify supplied values.
3. Do not judge accent.
4. Do not make clinical claims.
5. Do not criticize good metrics
   simply to create feedback.
6. Recommend at most one main
   speech focus.

Return structured speech analysis.
"""

        result = self.speech_llm.invoke(
            prompt
        )


        return {

            "speech_analysis":
                result.model_dump()
        }


    # =====================================================
    # NODE 4
    # CONTENT SPECIALIST
    # =====================================================

    def _analyze_content(
        self,
        state: CoachState
    ):

        print(
            "Agent: analyzing presentation content..."
        )

        evidence = state[
            "evidence"
        ]

        transcript = (
            evidence.get(
                "transcript",
                ""
            )
            or ""
        )

        word_count = (
            evidence.get(
                "word_count",
                0
            )
            or 0
        )


        if (
            not transcript.strip()
            or word_count < 20
        ):

            return {

                "content_analysis": {

                    "available":
                        False,

                    "summary":
                        "Not enough recognized speech "
                        "was available for reliable "
                        "content-level coaching.",

                    "strengths":
                        [],

                    "improvements":
                        []
                }
            }


        prompt = f"""
You are the Presentation Content Specialist
inside PresentAI Coach.

Analyze this recognized transcript:

TRANSCRIPT:
{transcript}

WORD COUNT:
{word_count}

Evaluate it ONLY as spoken presentation content.

Focus on:

- clarity
- conciseness
- organization
- transitions
- repetition
- understandable main idea
- whether phrasing sounds natural aloud

IMPORTANT:

1. Do not fact-check the speaker.
2. Do not invent missing context.
3. Do not rewrite the full speech.
4. Do not harshly judge grammar.
5. Speech recognition may contain
   transcription errors.
6. Keep feedback proportional to
   the transcript length.
7. Do not make claims that cannot
   be supported by the transcript.

Return structured content analysis.
"""

        result = self.content_llm.invoke(
            prompt
        )


        return {

            "content_analysis":
                result.model_dump()
        }


    # =====================================================
    # NODE 5
    # COACHING STRATEGY / PRIORITIZATION
    # =====================================================

    def _prioritize(
        self,
        state: CoachState
    ):

        print(
            "Agent: deciding coaching priorities..."
        )


        evidence = state[
            "evidence"
        ]

        delivery = state[
            "delivery_analysis"
        ]

        speech = state[
            "speech_analysis"
        ]

        content = state[
            "content_analysis"
        ]


        prompt = f"""
You are the Coaching Strategy Agent
inside PresentAI Coach.

You must reason across multiple specialist
analyses and decide what the presenter
should practice NEXT.

OBJECTIVE EVIDENCE:
{evidence}

VISUAL SPECIALIST:
{delivery}

SPEECH SPECIALIST:
{speech}

CONTENT SPECIALIST:
{content}

Your job is to choose the most
important improvement priorities.

RULES:

1. Analyzer measurements are authoritative.
2. Never invent numbers.
3. Never contradict objective scores.
4. Normally prioritize the weakest
   meaningful dimension.
5. Also consider unusually high-impact
   issues such as extremely fast delivery.
6. Strong behaviors should be preserved.
7. Select no more than THREE priorities.
8. Do not overwhelm the presenter.
9. Every priority must contain:
   - why it matters
   - one practical action
10. No medical posture claims.
11. If speech was insufficient,
    do not create speech priorities.

Rank priorities from most important
to least important.

Return structured priority analysis.
"""

        result = self.priority_llm.invoke(
            prompt
        )


        return {

            "priority_analysis":
                result.model_dump()
        }


    # =====================================================
    # NODE 6
    # NATURAL AI COACH
    # =====================================================

    def _generate_coaching(
        self,
        state: CoachState
    ):

        print(
            "Agent: generating personalized coaching..."
        )


        evidence = state[
            "evidence"
        ]

        delivery = state[
            "delivery_analysis"
        ]

        speech = state[
            "speech_analysis"
        ]

        content = state[
            "content_analysis"
        ]

        priorities = state[
            "priority_analysis"
        ]


        prompt = f"""
You are PresentAI Coach.

Act as an experienced, constructive,
natural presentation coach.

You are coaching ONE specific attempt.

OBJECTIVE EVIDENCE:
{evidence}

VISUAL ANALYSIS:
{delivery}

SPEECH ANALYSIS:
{speech}

CONTENT ANALYSIS:
{content}

COACHING STRATEGY:
{priorities}

The presenter should clearly understand:

- what worked
- what reduced impact
- why the main issue matters
- what to practice next
- which good behaviors to preserve

STYLE:

- natural
- supportive
- specific
- concise
- professional
- human-like
- not robotic
- not a dashboard
- not overly enthusiastic
- no repetitive advice

CRITICAL RULES:

1. Never invent scores or measurements.
2. Never contradict analyzer evidence.
3. Do not diagnose posture or health.
4. Never criticize a strong metric
   without evidence.
5. Do not create more than
   three improvement priorities.
6. If content evidence is limited,
   explicitly keep content feedback limited.
7. Make the next attempt plan practical.

Return structured coaching.
"""

        result = self.coaching_llm.invoke(
            prompt
        )


        return {

            "draft_coaching":
                result.model_dump()
        }


    # =====================================================
    # NODE 7
    # CRITIC AGENT
    # =====================================================

    def _critique(
        self,
        state: CoachState
    ):

        print(
            "Agent: reviewing coaching quality..."
        )


        evidence = state[
            "evidence"
        ]

        draft = state[
            "draft_coaching"
        ]


        prompt = f"""
You are the Quality Control Critic
inside PresentAI Coach.

OBJECTIVE EVIDENCE:
{evidence}

DRAFT COACHING:
{draft}

Audit the draft carefully.

Check for:

- invented numbers
- contradiction of scores
- praising a weak area incorrectly
- criticizing a strong area incorrectly
- unsupported statements
- too many priorities
- repetitive advice
- generic coaching
- medical posture claims
- unsupported transcript conclusions
- recommendations that do not follow
  from the evidence

Approval should be TRUE only if
the coaching is grounded and useful.

If revision is needed, provide
precise revision instructions.

Do NOT rewrite the coaching yourself.

Return structured critique.
"""

        result = self.critique_llm.invoke(
            prompt
        )


        return {

            "critique":
                result.model_dump()
        }


    # =====================================================
    # NODE 8
    # FINALIZER / REVISION
    # =====================================================

    def _finalize(
        self,
        state: CoachState
    ):

        print(
            "Agent: finalizing coaching..."
        )


        critique = state[
            "critique"
        ]

        draft = state[
            "draft_coaching"
        ]


        if critique.get(
            "approved",
            False
        ):

            print(
                "Agent critic: coaching approved."
            )

            return {

                "final_coaching":
                    draft
            }


        print(
            "Agent critic: revision required."
        )


        evidence = state[
            "evidence"
        ]


        prompt = f"""
You are the Final Coaching Editor
inside PresentAI Coach.

The critic rejected the first draft.

OBJECTIVE EVIDENCE:
{evidence}

ORIGINAL DRAFT:
{draft}

CRITIC FEEDBACK:
{critique}

Revise the coaching so that every
claim is grounded in objective evidence.

RULES:

1. Never invent measurements.
2. Never contradict supplied scores.
3. Do not make medical claims.
4. Keep no more than three
   improvement priorities.
5. Preserve strong behaviors.
6. Remove unsupported claims.
7. Remove repetitive advice.
8. Keep the tone natural,
   concise and coach-like.
9. Produce practical next steps.

Return the corrected structured
coaching response.
"""


        result = self.coaching_llm.invoke(
            prompt
        )


        return {

            "final_coaching":
                result.model_dump()
        }