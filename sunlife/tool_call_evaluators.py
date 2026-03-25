"""
Tool call evaluators - deterministic metrics (coverage, F1, arguments, trajectory)
plus optional deepeval LLM-judge evaluation.
Each evaluator returns a Langfuse Evaluation with a 0.0–1.0 score and a reason.
"""

from typing import Any

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langfuse.experiment import Evaluation


def _calculate_edit_distance(seq1: list[str], seq2: list[str]) -> int:
    """Calculate Levenshtein edit distance between two sequences."""
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    # Create DP table
    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    # Initialize base cases
    for i in range(len(seq1) + 1):
        dp[i][0] = i
    for j in range(len(seq2) + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    return dp[len(seq1)][len(seq2)]


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


def evaluate_tool_calls_trajectory(
    actual_tool_calls: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
) -> Evaluation:
    """
    Evaluates the sequence/order of tool calls using edit distance.
    Measures how similar the actual tool call sequence is to the expected sequence.

    Score: Normalized sequence similarity (0.0 to 1.0)
           1.0 = perfect match, 0.0 = completely different
    """
    if not expected_tool_calls:
        score = 1.0 if not actual_tool_calls else 0.0
        return Evaluation(
            name="tool_calls_trajectory",
            value=score,
            comment="No expected tool calls" if not actual_tool_calls else "Used tools when none expected",
        )

    # Extract tool name sequences
    expected_sequence = [tc["name"] for tc in expected_tool_calls]
    actual_sequence = [tc["name"] for tc in actual_tool_calls]

    # Calculate edit distance
    edit_distance = _calculate_edit_distance(expected_sequence, actual_sequence)

    # Normalize by the length of the longer sequence
    max_length = max(len(expected_sequence), len(actual_sequence))

    # Similarity score: 1.0 - (edit_distance / max_length)
    # Edit distance of 0 = perfect match (score 1.0)
    # Edit distance of max_length = completely different (score 0.0)
    score = 1.0 - (edit_distance / max_length) if max_length > 0 else 1.0

    # Build comment with details
    comment_parts = [f"Edit distance: {edit_distance}/{max_length}"]

    if edit_distance > 0:
        # Show sequence comparison
        expected_str = " → ".join(expected_sequence)
        actual_str = " → ".join(actual_sequence)
        comment_parts.append(f"Expected: {expected_str}")
        comment_parts.append(f"Actual: {actual_str}")

    return Evaluation(
        name="tool_calls_trajectory",
        value=score,
        comment=" | ".join(comment_parts),
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


def evaluate_plan_quality_llm_judge(
    question: str,
    plan: dict[str, Any],
    available_tools: list[str],
) -> Evaluation:
    """
    Evaluates the quality of the agent's research plan using LLM as judge.

    Assesses whether the plan is logical, appropriate for the question,
    and makes good use of available tools.

    Score: 0.0 to 1.0 based on plan quality
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    # Format plan steps for evaluation
    steps_text = "\n".join(
        f"{i+1}. {step['description']} (type: {step['step_type']}, expected: {step['expected_output']})"
        for i, step in enumerate(plan.get("steps", []))
    )

    plan_text = f"""
Question: {question}

Plan Reasoning: {plan.get('reasoning', 'N/A')}

Plan Steps:
{steps_text}

Available Tools: {', '.join(available_tools)}
"""

    # Create GEval metric for plan quality
    plan_quality_metric = GEval(
        name="Plan Quality",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Assess if the plan steps are logical and well-structured",
            "Check if the plan makes appropriate use of the available tools",
            "Evaluate if the plan is comprehensive enough to answer the question",
            "Verify that the plan steps have clear expected outputs",
            "Check if the reasoning explains why this plan was chosen",
        ],
        threshold=0.5,
    )

    test_case = LLMTestCase(
        input=f"Question: {question}\nAvailable Tools: {', '.join(available_tools)}",
        actual_output=plan_text,
    )

    plan_quality_metric.measure(test_case)

    return Evaluation(
        name="plan_quality_llm_judge",
        value=plan_quality_metric.score,
        comment=plan_quality_metric.reason or "Plan quality evaluated by LLM judge",
    )
