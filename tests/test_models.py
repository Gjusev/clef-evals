"""Tests for typed models: payload builders, response parsing, pricing."""

from __future__ import annotations

import pytest

from clef_evals.exceptions import ClefResponseError
from clef_evals.models import (
    INPUT_PRICE_PER_MILLION_TOKENS,
    Choice,
    Noul,
    Score,
    Usage,
    build_payload,
    parse_clef_response,
)


class TestQuestionBuilders:
    def test_noul_payload_minimal(self) -> None:
        assert Noul(instructions="Is it urgent?").to_payload() == {
            "type": "noul",
            "instructions": "Is it urgent?",
        }

    def test_noul_payload_with_criteria(self) -> None:
        payload = Noul(
            instructions="Is it urgent?",
            criteria_true="needs action now",
            criteria_false="can wait",
        ).to_payload()
        assert payload["criteria"] == {"true": "needs action now", "false": "can wait"}

    def test_choice_payload_includes_criteria(self) -> None:
        question = Choice(instructions="Pick a team", criteria={"billing": "money", "tech": "bugs"})
        assert question.to_payload() == {
            "type": "choice",
            "instructions": "Pick a team",
            "criteria": {"billing": "money", "tech": "bugs"},
        }

    def test_choice_rejects_single_option(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            Choice(instructions="x", criteria={"only": "one"})

    def test_choice_rejects_over_255_options(self) -> None:
        with pytest.raises(ValueError, match="at most 255"):
            Choice(instructions="x", criteria={f"o{i}": "" for i in range(256)})

    def test_score_payload_keeps_order(self) -> None:
        payload = Score(instructions="Rate severity", criteria=["low", "mid", "high"]).to_payload()
        assert payload["criteria"] == ["low", "mid", "high"]

    def test_score_rejects_bad_level_counts(self) -> None:
        with pytest.raises(ValueError):
            Score(instructions="x", criteria=["only"])
        with pytest.raises(ValueError):
            Score(instructions="x", criteria=[str(i) for i in range(11)])


class TestBuildPayload:
    def test_payload_includes_model_state_and_questions(self) -> None:
        payload = build_payload(
            "clef-flash",
            "the state",
            {"q": Noul(instructions="yes?")},
        )
        assert payload["model"] == "clef-flash"
        assert payload["state"] == "the state"
        assert payload["questions"]["q"] == {"type": "noul", "instructions": "yes?"}
        assert "images" not in payload

    def test_payload_rejects_empty_questions(self) -> None:
        with pytest.raises(ValueError, match="1 and 64"):
            build_payload("clef", "s", {})

    def test_payload_rejects_over_64_questions(self) -> None:
        questions = {f"q{i}": Noul(instructions="y?") for i in range(65)}
        with pytest.raises(ValueError, match="1 and 64"):
            build_payload("clef", "s", questions)

    def test_payload_passes_images_through(self) -> None:
        images = [{"content_type": "image/png", "base64": "aGk="}]
        payload = build_payload("clef", "s", {"q": Noul(instructions="y?")}, images=images)
        assert payload["images"] == images


class TestParseResponse:
    def test_parses_choice_answer(self) -> None:
        data = {
            "model": "@cf/cloudflare/clef",
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "tech": 0.1},
                    "confidence": 0.9,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        response = parse_clef_response(data, latency_ms=12.5)
        assert response.model == "@cf/cloudflare/clef"
        assert response.usage.input_tokens == 10
        assert response.latency_ms == 12.5
        answer = response.answers["decision"]
        assert answer.choice == "billing"
        assert answer.probabilities == {"billing": 0.9, "tech": 0.1}

    def test_parses_noul_answer(self) -> None:
        data = {
            "model": "@cf/cloudflare/clef",
            "answers": {"is_true": {"type": "noul", "noul": 0.87}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        response = parse_clef_response(data)
        assert response.answers["is_true"].probability == 0.87

    def test_parses_score_answer_with_legend(self) -> None:
        data = {
            "model": "@cf/cloudflare/clef",
            "answers": {
                "severity": {
                    "type": "score",
                    "score": 1.82,
                    "legend": {"0": "none", "1": "minor", "2": "major"},
                    "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
                    "confidence": 0.6,
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        response = parse_clef_response(data)
        answer = response.answers["severity"]
        assert answer.score == 1.82
        assert answer.legend == {0: "none", 1: "minor", 2: "major"}

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda d: d.pop("answers"),
            lambda d: d.pop("usage"),
            lambda d: d.pop("model"),
            lambda d: d["answers"]["decision"].pop("choice"),
            lambda d: d["usage"].pop("input_tokens"),
            lambda d: d["answers"].clear(),
            lambda d: d.update(answers=None),
        ],
    )
    def test_malformed_responses_raise_response_error(self, mutation) -> None:
        data = {
            "model": "@cf/cloudflare/clef",
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 1.0},
                    "confidence": 1.0,
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mutation(data)
        with pytest.raises(ClefResponseError):
            parse_clef_response(data)

    def test_unknown_answer_type_raises(self) -> None:
        data = {
            "model": "m",
            "answers": {"q": {"type": "essay", "text": "no"}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with pytest.raises(ClefResponseError, match="unknown type"):
            parse_clef_response(data)

    def test_non_object_result_raises(self) -> None:
        with pytest.raises(ClefResponseError):
            parse_clef_response(["not", "an", "object"])


class TestUsagePricing:
    def test_input_cost_uses_published_price(self) -> None:
        assert INPUT_PRICE_PER_MILLION_TOKENS == 0.24
        assert Usage(1_000_000, 0).input_cost_usd == 0.24
        assert Usage(120, 8).input_cost_usd == 120 * 0.24 / 1_000_000
