"""Typed request and response models for the Clef decision API.

The builders in this module produce exactly the payload documented at
https://developers.cloudflare.com/workers-ai/models/clef/ and the parsers
validate responses strictly, raising :class:`clef_evals.exceptions.ClefResponseError`
on anything malformed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .exceptions import ClefResponseError

#: Published input-token price for the hosted Clef models, in USD per million tokens.
INPUT_PRICE_PER_MILLION_TOKENS = 0.24


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is the probability that the answer is yes.

    Attributes:
        instructions: The yes/no question to evaluate.
        criteria_true: Optional description of what a yes (value near 1) means.
        criteria_false: Optional description of what a no (value near 0) means.
    """

    instructions: str
    criteria_true: str | None = None
    criteria_false: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON fragment for this question."""
        payload: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        criteria: dict[str, str] = {}
        if self.criteria_true is not None:
            criteria["true"] = self.criteria_true
        if self.criteria_false is not None:
            criteria["false"] = self.criteria_false
        if criteria:
            payload["criteria"] = criteria
        return payload


@dataclass(frozen=True)
class Choice:
    """Pick exactly one option from a set you define.

    Attributes:
        instructions: What the model should decide.
        criteria: Map of option id to its description (2 to 255 options).
    """

    instructions: str
    criteria: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("choice questions need at least 2 options")
        if len(self.criteria) > 255:
            raise ValueError("choice questions allow at most 255 options")

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON fragment for this question."""
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    """Rate the state on an ordered rubric of levels.

    Attributes:
        instructions: What the model should rate.
        criteria: Ordered level descriptions, lowest first (2 to 10 levels).
    """

    instructions: str
    criteria: list[str]

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("score questions need at least 2 levels")
        if len(self.criteria) > 10:
            raise ValueError("score questions allow at most 10 levels")

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON fragment for this question."""
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score


@dataclass(frozen=True)
class Usage:
    """Token usage reported by the API for one request.

    Attributes:
        input_tokens: Tokens consumed by the request state and schema.
        output_tokens: Tokens produced for the structured answers.
    """

    input_tokens: int
    output_tokens: int

    @property
    def input_cost_usd(self) -> float:
        """Cost of the input tokens at the published price.

        Output-token pricing is not published by Cloudflare, so this covers
        the input side only; ``output_tokens`` is reported for reference.
        """
        return self.input_tokens * INPUT_PRICE_PER_MILLION_TOKENS / 1_000_000


@dataclass(frozen=True)
class NoulAnswer:
    """Answer to a yes/no question: the probability that the answer is yes."""

    question_id: str
    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """Answer to a choice question with the full probability distribution."""

    question_id: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    """Answer to a score question: probability-weighted level index."""

    question_id: str
    score: float
    legend: dict[int, str]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class ClefResponse:
    """Parsed Clef API response for one request.

    Attributes:
        model: Model that performed the evaluation (as reported by the API).
        answers: Parsed answers keyed by question id.
        usage: Token usage for the request.
        latency_ms: Wall-clock round-trip time measured by the client.
    """

    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ClefResponseError(f"response is missing required field {key!r}")
    return data[key]


def _parse_probabilities(raw: Any, question_id: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ClefResponseError(f"answer {question_id!r} has non-object probabilities")
    probabilities: dict[str, float] = {}
    for option, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ClefResponseError(f"answer {question_id!r} has non-numeric probability for {option!r}")
        probabilities[str(option)] = float(value)
    return probabilities


def _parse_answer(question_id: str, raw: Any) -> Answer:
    if not isinstance(raw, Mapping):
        raise ClefResponseError(f"answer {question_id!r} is not an object")
    answer_type = raw.get("type")
    if answer_type == "noul":
        probability = _require(raw, "noul")
        if not isinstance(probability, (int, float)):
            raise ClefResponseError(f"answer {question_id!r} has non-numeric noul probability")
        return NoulAnswer(question_id=question_id, probability=float(probability))
    if answer_type == "choice":
        probabilities = _parse_probabilities(_require(raw, "probabilities"), question_id)
        confidence = _require(raw, "confidence")
        if not isinstance(confidence, (int, float)):
            raise ClefResponseError(f"answer {question_id!r} has non-numeric confidence")
        return ChoiceAnswer(
            question_id=question_id,
            choice=str(_require(raw, "choice")),
            probabilities=probabilities,
            confidence=float(confidence),
        )
    if answer_type == "score":
        legend_raw = _require(raw, "legend")
        if not isinstance(legend_raw, Mapping):
            raise ClefResponseError(f"answer {question_id!r} has non-object legend")
        legend = {int(level): str(description) for level, description in legend_raw.items()}
        score_value = _require(raw, "score")
        if not isinstance(score_value, (int, float)):
            raise ClefResponseError(f"answer {question_id!r} has non-numeric score")
        probabilities = _parse_probabilities(_require(raw, "probabilities"), question_id)
        confidence = _require(raw, "confidence")
        if not isinstance(confidence, (int, float)):
            raise ClefResponseError(f"answer {question_id!r} has non-numeric confidence")
        return ScoreAnswer(
            question_id=question_id,
            score=float(score_value),
            legend=legend,
            probabilities=probabilities,
            confidence=float(confidence),
        )
    raise ClefResponseError(f"answer {question_id!r} has unknown type {answer_type!r}")


def parse_clef_response(data: Mapping[str, Any], latency_ms: float = 0.0) -> ClefResponse:
    """Validate and parse a ``result`` payload returned by the Clef API.

    Args:
        data: The ``result`` object from the Cloudflare envelope.
        latency_ms: Wall-clock latency measured by the caller.

    Returns:
        A typed :class:`ClefResponse`.

    Raises:
        ClefResponseError: If required fields are missing or malformed.
    """
    if not isinstance(data, Mapping):
        raise ClefResponseError("result payload is not an object")
    answers_raw = _require(data, "answers")
    if not isinstance(answers_raw, Mapping) or not answers_raw:
        raise ClefResponseError("answers must be a non-empty object")
    answers = {str(qid): _parse_answer(str(qid), raw) for qid, raw in answers_raw.items()}
    usage_raw = _require(data, "usage")
    if not isinstance(usage_raw, Mapping):
        raise ClefResponseError("usage is not an object")
    usage = Usage(
        input_tokens=int(_require(usage_raw, "input_tokens")),
        output_tokens=int(_require(usage_raw, "output_tokens")),
    )
    return ClefResponse(
        model=str(_require(data, "model")),
        answers=answers,
        usage=usage,
        latency_ms=latency_ms,
    )


def build_payload(
    model_selector: str,
    state: Any,
    questions: Mapping[str, Question],
    images: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Assemble the request body for the Clef API.

    Args:
        model_selector: ``"clef"`` or ``"clef-flash"`` (see :meth:`ClefConfig.model_selector`).
        state: Content to evaluate: a string or structured JSON data.
        questions: Question builders keyed by question id (1 to 64 questions).
        images: Optional list of pre-built image fragments.

    Returns:
        The JSON-serializable request body.
    """
    if not 1 <= len(questions) <= 64:
        raise ValueError("between 1 and 64 questions are required")
    payload: dict[str, Any] = {
        "model": model_selector,
        "state": state,
        "questions": {qid: question.to_payload() for qid, question in questions.items()},
    }
    if images:
        payload["images"] = images
    return payload
