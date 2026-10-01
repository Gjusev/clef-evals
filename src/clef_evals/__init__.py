"""clef-evals: Calibration-first evaluation toolkit for Cloudflare's Clef."""

import os
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np


def ece(confidences: list[float], correct: list[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confs = np.array(confidences)
    corr = np.array([1.0 if c else 0.0 for c in correct])
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = corr[mask].mean()
        ece_val += (mask.sum() / len(confs)) * abs(bin_acc - bin_conf)
    return float(ece_val)


def brier_score(confidences: list[float], correct: list[bool]) -> float:
    """Brier score: mean squared error of probabilistic predictions."""
    confs = np.array(confidences)
    corr = np.array([1.0 if c else 0.0 for c in correct])
    return float(np.mean((confs - corr) ** 2))


@dataclass
class ClefEvalResult:
    accuracy: float
    ece: float
    brier: float
    n_samples: int
    correct: list[bool]
    confidences: list[float]


@dataclass
class ClefJudge:
    """Run Clef as an LLM-as-judge over an eval set."""

    account_id: str = os.environ.get("CLEF_ACCOUNT_ID", "")
    api_token: str = os.environ.get("CLEF_API_TOKEN", "")
    model: str = "@cf/cloudflare/clef-flash"
    base_url: str = "https://api.cloudflare.com/client/v4"
    timeout: float = 60.0

    def __post_init__(self):
        if not self.account_id:
            raise ValueError("Set CLEF_ACCOUNT_ID")

    def judge_choice(self, state: str, question: str, choices: list[str]) -> tuple[str, float]:
        """Ask Clef to choose one option. Returns (choice, confidence)."""
        payload = {
            "state": state,
            "questions": {
                question: {
                    "type": "choice",
                    "choices": choices,
                },
                "confidence": {
                    "type": "score",
                    "context": "How confident are you?",
                    "levels": ["very_low", "low", "medium", "high", "very_high"],
                },
            },
        }
        resp = httpx.post(
            f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        choice = result.get(question, choices[-1])
        conf_map = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "very_high": 0.9}
        confidence = conf_map.get(result.get("confidence", "low"), 0.3)
        return choice, confidence

    def evaluate(
        self,
        eval_set: list[dict],
        state_key: str = "state",
        question_key: str = "question",
        choices_key: str = "choices",
        gold_key: str = "gold",
    ) -> ClefEvalResult:
        """Run judge over an eval set and compute calibration metrics."""
        correct_flags, confidence_list = [], []
        for item in eval_set:
            pred, conf = self.judge_choice(
                item[state_key], item[question_key], item[choices_key]
            )
            is_correct = pred == item[gold_key]
            correct_flags.append(is_correct)
            confidence_list.append(conf)

        accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
        return ClefEvalResult(
            accuracy=accuracy,
            ece=ece(confidence_list, correct_flags),
            brier=brier_score(confidence_list, correct_flags),
            n_samples=len(correct_flags),
            correct=correct_flags,
            confidences=confidence_list,
        )


__version__ = "0.1.0"
__all__ = ["ClefJudge", "ClefEvalResult", "ece", "brier_score"]
