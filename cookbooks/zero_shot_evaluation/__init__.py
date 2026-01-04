# -*- coding: utf-8 -*-
"""Zero-Shot Evaluation module for comparing models and agent pipelines.

Usage:
    # CLI
    python -m cookbooks.zero_shot_evaluation --config config.yaml

    # Python
    from cookbooks.zero_shot_evaluation import ZeroShotEvaluator
    evaluator = ZeroShotEvaluator.from_config("config.yaml")
    result = await evaluator.evaluate()
"""

from cookbooks.zero_shot_evaluation.core.config import load_config
from cookbooks.zero_shot_evaluation.core.evaluator import EvaluationResult, ZeroShotEvaluator
from cookbooks.zero_shot_evaluation.core.query_generator import QueryGenerator
from cookbooks.zero_shot_evaluation.core.response_collector import ResponseCollector
from cookbooks.zero_shot_evaluation.core.rubric_generator import RubricGenerator
from cookbooks.zero_shot_evaluation.core.schema import (
    EvaluationConfig,
    OpenAIEndpoint,
    QueryGenerationConfig,
    TaskConfig,
    ZeroShotConfig,
)

__all__ = [
    # Config
    "ZeroShotConfig",
    "TaskConfig",
    "OpenAIEndpoint",
    "QueryGenerationConfig",
    "EvaluationConfig",
    "load_config",
    # Components
    "QueryGenerator",
    "ResponseCollector",
    "RubricGenerator",
    # Evaluator
    "ZeroShotEvaluator",
    "EvaluationResult",
]

