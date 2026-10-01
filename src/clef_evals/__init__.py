"""clef-evals: Calibration-first evaluation toolkit for Cloudflare's Clef.

Judge cheap, audit confidence. Run Cloudflare's `Clef decision models
<https://developers.cloudflare.com/workers-ai/models/clef/>`_ as an
LLM-as-judge over eval sets and gate CI on accuracy *and* calibration
(Expected Calibration Error, Brier score).
"""

from __future__ import annotations

from .client import ClefClient
from .config import ClefConfig
from .exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
    ConfigurationError,
)
from .judge import AsyncClefJudge, ChoiceDecision, ClefJudge, EvalItem, load_eval_set
from .metrics import ClefEvalResult, EvalResult, brier_multiclass, brier_score, ece
from .models import (
    Answer,
    Choice,
    ChoiceAnswer,
    ClefResponse,
    Noul,
    Score,
    ScoreAnswer,
    Usage,
)

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "AsyncClefJudge",
    "Choice",
    "ChoiceAnswer",
    "ChoiceDecision",
    "ClefAPIError",
    "ClefAuthError",
    "ClefClient",
    "ClefConfig",
    "ClefError",
    "ClefEvalResult",
    "ClefJudge",
    "ClefNetworkError",
    "ClefRateLimitError",
    "ClefResponse",
    "ClefResponseError",
    "ClefServerError",
    "ClefTimeoutError",
    "ConfigurationError",
    "EvalItem",
    "EvalResult",
    "Noul",
    "Score",
    "ScoreAnswer",
    "Usage",
    "__version__",
    "brier_multiclass",
    "brier_score",
    "ece",
    "load_eval_set",
]
