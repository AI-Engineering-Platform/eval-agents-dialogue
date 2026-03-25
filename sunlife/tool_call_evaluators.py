"""
Tool call evaluators - deterministic metrics (coverage, F1, arguments)
plus optional deepeval LLM-judge evaluation.
Each evaluator returns a Langfuse Evaluation with a 0.0–1.0 score and a reason.
"""

from typing import Any

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langfuse.experiment import Evaluation


def _convert_tool_calls_to_deepeval_format(tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert tool calls from agent format to deepeval ToolCall format."""
    return [
        ToolCall(
            name=tc["name"],
            arguments=tc.get("args", {}),
        )
        for tc in tool_calls
    ]


def evaluate_tool_calls_coverage(
    actual_tool_calls: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
) -> Evaluation:
    """
    Checks if all expected tool types were called at least once.
    Ignores number of calls and arguments.

    Score: Percentage of expected tool types that were called (0.0 to 1.0)
    """
    if not expected_tool_calls:
        return Evaluation(
            name="tool_calls_coverage",
            value=1.0,
            comment="No expected tool calls",
        )

    expected_tool_types = set(tc["name"] for tc in expected_tool_calls)
    actual_tool_types = set(tc["name"] for tc in actual_tool_calls)

    covered_tools = expected_tool_types.intersection(actual_tool_types)
    coverage = len(covered_tools) / len(expected_tool_types)

    missing_tools = expected_tool_types - actual_tool_types
    extra_tools = actual_tool_types - expected_tool_types

    reason_parts = [f"Coverage: {coverage:.1%}"]
    if missing_tools:
        reason_parts.append(f"Missing: {', '.join(missing_tools)}")
    if extra_tools:
        reason_parts.append(f"Extra: {', '.join(extra_tools)}")

    return Evaluation(
        name="tool_calls_coverage",
        value=coverage,
        comment=" | ".join(reason_parts),
    )


def evaluate_tool_calls_f1(
    actual_tool_calls: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
) -> Evaluation:
    """
    Evaluates tool calls using precision and recall, computing F1 score.
    Compares each expected tool call against actual tool calls (1:1 matching).

    Score: F1 score (0.0 to 1.0)
    """
    if not expected_tool_calls:
        score = 1.0 if not actual_tool_calls else 0.0
        return Evaluation(
            name="tool_calls_f1",
            value=score,
            comment="No expected tool calls" if not actual_tool_calls else "Used tools when none expected",
        )

    # Match each expected tool call with an actual tool call
    matched_actual = set()
    matched_expected = 0

    for expected_tc in expected_tool_calls:
        expected_name = expected_tc["name"]

        # Find first unmatched actual tool call with same name
        for i, actual_tc in enumerate(actual_tool_calls):
            if i not in matched_actual and actual_tc["name"] == expected_name:
                matched_expected += 1
                matched_actual.add(i)
                break

    # Precision: what fraction of actual tool calls were correct
    precision = len(matched_actual) / len(actual_tool_calls) if actual_tool_calls else 0.0

    # Recall: what fraction of expected tool calls were found
    recall = matched_expected / len(expected_tool_calls) if expected_tool_calls else 0.0

    # F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return Evaluation(
        name="tool_calls_f1",
        value=f1,
        comment=f"Matched {matched_expected}/{len(expected_tool_calls)} expected, {len(matched_actual)}/{len(actual_tool_calls)} actual | P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}",
    )


def evaluate_tool_calls_arguments(
    actual_tool_calls: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
) -> Evaluation:
    """
    Evaluates if tool arguments match.
    Compares each expected tool call's arguments with actual tool calls.

    Score: Percentage of expected tool calls with matching arguments (0.0 to 1.0)
    """
    if not expected_tool_calls:
        return Evaluation(
            name="tool_calls_arguments",
            value=1.0,
            comment="No expected tool calls",
        )

    # Count how many expected tool calls have a matching actual tool call
    matching_calls = 0

    for expected_tc in expected_tool_calls:
        expected_name = expected_tc["name"]
        expected_args = expected_tc.get("args", {})

        # Find matching actual tool call
        for actual_tc in actual_tool_calls:
            if actual_tc["name"] != expected_name:
                continue

            actual_args = actual_tc.get("args", {})

            # Check if all expected arg keys match
            all_match = all(
                key in actual_args and actual_args[key] == value
                for key, value in expected_args.items()
            )

            if all_match:
                matching_calls += 1
                break  # Found a match for this expected call

    score = matching_calls / len(expected_tool_calls)

    return Evaluation(
        name="tool_calls_arguments",
        value=score,
        comment=f"Tool calls with matching args: {matching_calls}/{len(expected_tool_calls)} ({score:.1%})",
    )


def evaluate_tool_correctness_llm_judge(
    actual_tool_calls: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
) -> Evaluation:
    """
    Evaluates tool correctness using deepeval's ToolCorrectnessMetric with LLM judge.
    Provides a holistic evaluation of tool usage correctness.

    Score: 0.0 to 1.0 based on LLM judge evaluation
    """
    metric = ToolCorrectnessMetric(
        threshold=0.5,
        include_reason=True,
        async_mode=False,
    )

    test_case = LLMTestCase(
        input="",  # Not used for tool evaluation
        actual_output="",  # Not used for tool evaluation
        actual_tools_used=_convert_tool_calls_to_deepeval_format(actual_tool_calls),
        expected_tools=_convert_tool_calls_to_deepeval_format(expected_tool_calls),
    )

    metric.measure(test_case)

    return Evaluation(
        name="tool_correctness_llm_judge",
        value=metric.score,
        comment=metric.reason or "Tool correctness evaluated by LLM judge",
    )