"""Judge layer: run Clef over eval sets and measure accuracy + calibration."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ClefClient
from .config import ClefConfig
from .exceptions import ClefError
from .metrics import (
    EvalResult,
    brier_multiclass,
    brier_score,
    ece,
    latency_percentiles,
)
from .models import Choice, ChoiceAnswer, ClefResponse, Noul, NoulAnswer, Question, Usage

logger = logging.getLogger("clef_evals.judge")

#: Question id used for the single-choice decision question.
DEFAULT_CHOICE_QUESTION = "decision"
#: Question id used for the binary (noul) question.
DEFAULT_BINARY_QUESTION = "is_true"


class EvalItem(dict):
    """One evaluation example.

    Required keys: ``state``, ``instructions``, ``gold``.
    Choice items additionally need ``criteria`` (option -> description map) or
    ``choices`` (list of option ids, mapped to empty descriptions). Binary
    items use a boolean ``gold`` and no criteria.
    """

    @property
    def state(self) -> Any:
        """Content to evaluate."""
        return self["state"]

    @property
    def instructions(self) -> str:
        """What the model should decide for this item."""
        return self["instructions"]

    @property
    def gold(self) -> Any:
        """Gold label: option id (choice) or bool (binary)."""
        return self["gold"]

    def criteria(self) -> dict[str, str]:
        """Criteria map for choice items; empty for binary items."""
        if "criteria" in self:
            return {str(option): str(description) for option, description in self["criteria"].items()}
        if "choices" in self:
            return {str(option): "" for option in self["choices"]}
        return {}

    @property
    def is_binary(self) -> bool:
        """Whether this item is a yes/no (noul) item."""
        return isinstance(self["gold"], bool)


@dataclass(frozen=True)
class ChoiceDecision:
    """A single choice decision with its probability distribution.

    Attributes:
        choice: The chosen option id.
        probabilities: Probability per option.
        confidence: Confidence assigned to the chosen option.
        latency_ms: Wall-clock latency of the API call.
        usage: Token usage of the API call.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    usage: Usage


@dataclass(frozen=True)
class _ItemOutcome:
    """Internal per-item evaluation record."""

    correct: bool
    confidence: float
    prediction: Any
    gold: Any
    latency_ms: float
    distribution: dict[str, float] | None
    usage: Usage


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Load an eval set from a JSON array file or a JSONL file.

    Args:
        path: Dataset path (``.json`` or ``.jsonl``).

    Returns:
        List of validated :class:`EvalItem`.

    Raises:
        ValueError: If the file cannot be parsed or an item lacks required keys.
    """
    dataset_path = Path(path)
    text = dataset_path.read_text(encoding="utf-8")
    if dataset_path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw_items = json.loads(text)
    if not isinstance(raw_items, list):
        raise ValueError(f"{dataset_path}: dataset must be a JSON array or JSONL")
    items: list[EvalItem] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{dataset_path}: item {index} is not an object")
        missing = [key for key in ("state", "instructions", "gold") if key not in raw]
        if missing:
            raise ValueError(f"{dataset_path}: item {index} is missing keys {missing}")
        items.append(EvalItem(raw))
    return items


def _validate_items(eval_set: Sequence[Mapping[str, Any]]) -> None:
    for index, item in enumerate(eval_set):
        missing = [key for key in ("state", "instructions", "gold") if key not in item]
        if missing:
            raise ValueError(f"eval_set item {index} is missing keys {missing}")
        if not isinstance(item["gold"], bool) and "criteria" not in item and "choices" not in item:
            raise ValueError(
                f"eval_set item {index} needs 'criteria' (map) or 'choices' (list) for choice items"
            )


class ClefJudge:
    """Run Cloudflare Clef as an LLM-as-judge over eval sets (sync).

    Args:
        config: Pre-built :class:`clef_evals.config.ClefConfig`. When omitted,
            one is assembled from the environment via :meth:`ClefConfig.from_env`.
        client: Optional pre-built :class:`clef_evals.client.ClefClient`
            (tests inject a mock-transport client here).
        **overrides: Explicit config overrides applied on top of the config
            (``model``, ``timeout``, ``max_retries``, ...).

    Raises:
        ConfigurationError: If no valid configuration is available.
    """

    def __init__(
        self,
        config: ClefConfig | None = None,
        client: ClefClient | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = ClefConfig.from_env().with_overrides(**overrides)
        self.config = config
        self._owns_client = client is None
        self.client = client or ClefClient(config)

    def close(self) -> None:
        """Release the underlying HTTP client if owned."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> ClefJudge:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- single calls ----------------------------------------------------------

    def judge_choice(
        self,
        state: Any,
        instructions: str,
        criteria: Mapping[str, str],
        question_id: str = DEFAULT_CHOICE_QUESTION,
    ) -> ChoiceDecision:
        """Ask Clef to pick one option; return the decision with probabilities.

        Args:
            state: Content to evaluate.
            instructions: What the model should decide.
            criteria: Option id -> description map (2 to 255 options).
            question_id: Question id for the request.

        Returns:
            The :class:`ChoiceDecision` for the single question.

        Raises:
            ClefError: The API failed or returned an unexpected answer type.
        """
        response = self.client.run(state, {question_id: Choice(instructions=instructions, criteria=criteria)})
        return self._choice_decision(response, question_id)

    def judge_binary(
        self,
        state: Any,
        instructions: str,
        question_id: str = DEFAULT_BINARY_QUESTION,
    ) -> float:
        """Ask Clef a yes/no question; return the probability the answer is yes.

        Args:
            state: Content to evaluate.
            instructions: The yes/no question to evaluate.
            question_id: Question id for the request.

        Returns:
            Probability in [0, 1] that the answer is yes.

        Raises:
            ClefError: The API failed or returned an unexpected answer type.
        """
        response = self.client.run(state, {question_id: Noul(instructions=instructions)})
        answer = response.answers.get(question_id)
        if not isinstance(answer, NoulAnswer):
            raise ClefError(f"question {question_id!r} did not return a noul answer")
        return answer.probability

    def run_questions(
        self,
        state: Any,
        questions: Mapping[str, Question],
        images: list[dict[str, str]] | None = None,
    ) -> ClefResponse:
        """Send arbitrary typed questions in one request (full API surface).

        Args:
            state: Content to evaluate.
            questions: Typed question builders keyed by id.
            images: Optional embedded images.

        Returns:
            The full parsed :class:`clef_evals.models.ClefResponse`.
        """
        return self.client.run(state, questions, images)

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, eval_set: Sequence[Mapping[str, Any]]) -> EvalResult:
        """Evaluate a dataset sequentially and compute calibration metrics.

        Binary items (boolean ``gold``) are judged with a noul question;
        choice items with a choice question. Items whose API call fails are
        counted in ``failures`` and excluded from the metrics so one flaky
        call does not erase a run.

        Args:
            eval_set: Items (see :class:`EvalItem` for the expected shape).

        Returns:
            The aggregate :class:`EvalResult`.

        Raises:
            ValueError: If an item lacks required keys.
        """
        _validate_items(eval_set)
        outcomes: list[_ItemOutcome] = []
        failures = 0
        for index, item in enumerate(eval_set):
            try:
                outcomes.append(self._judge_item(item))
            except ClefError as error:
                failures += 1
                logger.warning("item %d failed: %s", index, error)
        return _aggregate(outcomes, failures=failures, model=self.config.model)

    # -- internals -------------------------------------------------------------

    def _judge_item(self, item: Mapping[str, Any]) -> _ItemOutcome:
        wrapped = EvalItem(item)
        if wrapped.is_binary:
            response = self.client.run(
                wrapped.state, {DEFAULT_BINARY_QUESTION: Noul(instructions=wrapped.instructions)}
            )
            answer = response.answers.get(DEFAULT_BINARY_QUESTION)
            if not isinstance(answer, NoulAnswer):
                raise ClefError(f"question {DEFAULT_BINARY_QUESTION!r} did not return a noul answer")
            predicted = answer.probability >= 0.5
            return _ItemOutcome(
                correct=predicted == wrapped.gold,
                confidence=max(answer.probability, 1.0 - answer.probability),
                prediction=predicted,
                gold=wrapped.gold,
                latency_ms=response.latency_ms,
                distribution=None,
                usage=response.usage,
            )
        decision = self.judge_choice(wrapped.state, wrapped.instructions, wrapped.criteria())
        return _ItemOutcome(
            correct=decision.choice == wrapped.gold,
            confidence=decision.probabilities.get(decision.choice, decision.confidence),
            prediction=decision.choice,
            gold=wrapped.gold,
            latency_ms=decision.latency_ms,
            distribution=decision.probabilities,
            usage=decision.usage,
        )

    def _choice_decision(self, response: ClefResponse, question_id: str) -> ChoiceDecision:
        answer = response.answers.get(question_id)
        if not isinstance(answer, ChoiceAnswer):
            raise ClefError(f"question {question_id!r} did not return a choice answer")
        return ChoiceDecision(
            choice=answer.choice,
            probabilities=answer.probabilities,
            confidence=answer.confidence,
            latency_ms=response.latency_ms,
            usage=response.usage,
        )


class AsyncClefJudge(ClefJudge):
    """Async variant of :class:`ClefJudge` built on ``httpx.AsyncClient``.

    ``evaluate`` fans out with a bounded semaphore so large datasets neither
    serialize nor stampede the API.
    """

    async def judge_choice_async(
        self,
        state: Any,
        instructions: str,
        criteria: Mapping[str, str],
        question_id: str = DEFAULT_CHOICE_QUESTION,
    ) -> ChoiceDecision:
        """Async version of :meth:`ClefJudge.judge_choice`."""
        response = await self.client.arun(
            state, {question_id: Choice(instructions=instructions, criteria=criteria)}
        )
        return self._choice_decision(response, question_id)

    async def judge_binary_async(
        self,
        state: Any,
        instructions: str,
        question_id: str = DEFAULT_BINARY_QUESTION,
    ) -> float:
        """Async version of :meth:`ClefJudge.judge_binary`."""
        response = await self.client.arun(state, {question_id: Noul(instructions=instructions)})
        answer = response.answers.get(question_id)
        if not isinstance(answer, NoulAnswer):
            raise ClefError(f"question {question_id!r} did not return a noul answer")
        return answer.probability

    async def evaluate(
        self,
        eval_set: Sequence[Mapping[str, Any]],
        concurrency: int = 5,
    ) -> EvalResult:
        """Evaluate a dataset concurrently; see :meth:`ClefJudge.evaluate`.

        Args:
            eval_set: Items to evaluate.
            concurrency: Maximum number of in-flight API calls (>= 1).

        Returns:
            The aggregate :class:`EvalResult` (input order preserved).
        """
        _validate_items(eval_set)
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def worker(index: int, item: Mapping[str, Any]) -> _ItemOutcome:
            async with semaphore:
                return await self._judge_item_async(item)

        results = await asyncio.gather(
            *(worker(i, item) for i, item in enumerate(eval_set)),
            return_exceptions=True,
        )

        outcomes: list[_ItemOutcome] = []
        failures = 0
        for index, result in enumerate(results):
            if isinstance(result, ClefError):
                failures += 1
                logger.warning("item %d failed: %s", index, result)
            elif isinstance(result, BaseException):
                raise result
            else:
                outcomes.append(result)
        return _aggregate(outcomes, failures=failures, model=self.config.model)

    # -- internals -------------------------------------------------------------

    async def _judge_item_async(self, item: Mapping[str, Any]) -> _ItemOutcome:
        wrapped = EvalItem(item)
        if wrapped.is_binary:
            response = await self.client.arun(
                wrapped.state, {DEFAULT_BINARY_QUESTION: Noul(instructions=wrapped.instructions)}
            )
            answer = response.answers.get(DEFAULT_BINARY_QUESTION)
            if not isinstance(answer, NoulAnswer):
                raise ClefError(f"question {DEFAULT_BINARY_QUESTION!r} did not return a noul answer")
            predicted = answer.probability >= 0.5
            return _ItemOutcome(
                correct=predicted == wrapped.gold,
                confidence=max(answer.probability, 1.0 - answer.probability),
                prediction=predicted,
                gold=wrapped.gold,
                latency_ms=response.latency_ms,
                distribution=None,
                usage=response.usage,
            )
        decision = await self.judge_choice_async(wrapped.state, wrapped.instructions, wrapped.criteria())
        return _ItemOutcome(
            correct=decision.choice == wrapped.gold,
            confidence=decision.probabilities.get(decision.choice, decision.confidence),
            prediction=decision.choice,
            gold=wrapped.gold,
            latency_ms=decision.latency_ms,
            distribution=decision.probabilities,
            usage=decision.usage,
        )


def _aggregate(
    outcomes: Sequence[_ItemOutcome],
    *,
    failures: int,
    model: str,
) -> EvalResult:
    """Combine per-item outcomes into the final :class:`EvalResult`."""
    correct = [outcome.correct for outcome in outcomes]
    confidences = [outcome.confidence for outcome in outcomes]
    predictions = [outcome.prediction for outcome in outcomes]
    golds = [outcome.gold for outcome in outcomes]
    latencies = [outcome.latency_ms for outcome in outcomes]
    distributions = [outcome.distribution for outcome in outcomes if outcome.distribution is not None]
    choice_golds = [
        str(outcome.gold) for outcome in outcomes if outcome.distribution is not None
    ]
    input_tokens = sum(outcome.usage.input_tokens for outcome in outcomes)
    output_tokens = sum(outcome.usage.output_tokens for outcome in outcomes)
    n = len(correct)
    accuracy = sum(correct) / n if n else 0.0
    mean_input_tokens = input_tokens / n if n else 0.0
    latencies_summary = latency_percentiles(latencies)
    return EvalResult(
        accuracy=accuracy,
        ece=ece(confidences, correct),
        brier=brier_score(confidences, correct),
        brier_multiclass=brier_multiclass(distributions, choice_golds),
        n_samples=n,
        correct=correct,
        confidences=confidences,
        predictions=predictions,
        golds=golds,
        latencies_ms=latencies,
        latency_p50=latencies_summary["p50"],
        latency_p95=latencies_summary["p95"],
        latency_p99=latencies_summary["p99"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_tokens * Usage(1, 0).input_cost_usd,
        cost_usd_per_1k_calls=mean_input_tokens * 1000 * Usage(1, 0).input_cost_usd,
        model=model,
        failures=failures,
    )


__all__ = [
    "AsyncClefJudge",
    "ChoiceDecision",
    "ClefJudge",
    "EvalItem",
    "EvalResult",
    "load_eval_set",
]
