"""
Langfuse-compatible wrapper evaluators.
These wrap the core evaluators from evaluators to match the Langfuse signature.
"""

from typing import Any

from langfuse.experiment import Evaluation

from .evaluators import (
    compute_composite_score,
    evaluate_tool_calls_arguments,
    evaluate_tool_calls_coverage,
    evaluate_tool_calls_f1,
    evaluate_tool_calls_trajectory,
)


def evaluate_coverage(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """Evaluate tool call coverage."""
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])
    return evaluate_tool_calls_coverage(actual_tool_calls, expected_tool_calls)


def evaluate_f1(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """Evaluate tool call F1 score."""
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])
    return evaluate_tool_calls_f1(actual_tool_calls, expected_tool_calls)


def evaluate_arguments(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """Evaluate tool call arguments."""
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])
    return evaluate_tool_calls_arguments(actual_tool_calls, expected_tool_calls)


def evaluate_trajectory(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """Evaluate tool call sequence/trajectory."""
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])
    return evaluate_tool_calls_trajectory(actual_tool_calls, expected_tool_calls)


def evaluate_composite(
    *,
    evaluations: list[Evaluation],
    **kwargs: Any,
) -> Evaluation:
    """Compute weighted composite score from all evaluator results."""
    return compute_composite_score(evaluations)
