"""
Deterministic tool call evaluators: coverage, F1, arguments, trajectory.
Each evaluator returns a Langfuse Evaluation with a 0.0–1.0 score and a reason.
"""

from pathlib import Path
from typing import Any

from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
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
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[len(seq1)][len(seq2)]


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
            comment=(
                "No expected tool calls"
                if not actual_tool_calls
                else "Used tools when none expected"
            ),
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
    precision = (
        len(matched_actual) / len(actual_tool_calls) if actual_tool_calls else 0.0
    )

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


def _normalize_value(value: Any) -> Any:
    """Normalize a value for fuzzy comparison."""
    if isinstance(value, str):
        # Lowercase, strip whitespace, remove common punctuation
        return value.lower().strip().rstrip(".,!?;:")
    elif isinstance(value, (int, float)):
        return value
    elif isinstance(value, bool):
        return value
    elif value is None:
        return None
    else:
        # For complex types, convert to normalized string
        return str(value).lower().strip()


def _values_match_fuzzy(actual_value: Any, expected_value: Any) -> bool:
    """Check if two values match with fuzzy comparison."""
    # Normalize both values
    actual_norm = _normalize_value(actual_value)
    expected_norm = _normalize_value(expected_value)

    # Exact match after normalization
    if actual_norm == expected_norm:
        return True

    # For strings, check if one contains the other (partial match)
    if isinstance(actual_norm, str) and isinstance(expected_norm, str):
        # Check substring match (either direction)
        if actual_norm in expected_norm or expected_norm in actual_norm:
            return True

        # Check token overlap for multi-word strings
        actual_tokens = set(actual_norm.split())
        expected_tokens = set(expected_norm.split())
        if actual_tokens and expected_tokens:
            # If >50% of tokens overlap, consider it a match
            overlap = len(actual_tokens & expected_tokens)
            min_tokens = min(len(actual_tokens), len(expected_tokens))
            if overlap / min_tokens > 0.5:
                return True

    # For numbers, allow small variations (within 1% for floats)
    if isinstance(actual_norm, (int, float)) and isinstance(
        expected_norm, (int, float)
    ):
        if expected_norm == 0:
            return actual_norm == 0
        rel_diff = abs(actual_norm - expected_norm) / abs(expected_norm)
        if rel_diff < 0.01:  # Within 1%
            return True

    return False


def evaluate_tool_calls_arguments(
    actual_tool_calls: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
) -> Evaluation:
    """
    Evaluates if tool arguments match using fuzzy comparison.
    Compares each expected tool call's arguments with actual tool calls.
    Uses normalized comparison with partial matching for strings.

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
    partial_matches = []

    for expected_tc in expected_tool_calls:
        expected_name = expected_tc["name"]
        expected_args = expected_tc.get("args", {})

        # Find matching actual tool call
        best_match_score = 0.0
        found_exact = False

        for actual_tc in actual_tool_calls:
            if actual_tc["name"] != expected_name:
                continue

            actual_args = actual_tc.get("args", {})

            # Check how many args match (fuzzy)
            if not expected_args:
                # No expected args, so it matches
                matching_calls += 1
                found_exact = True
                break

            matched_keys = sum(
                1
                for key, expected_val in expected_args.items()
                if key in actual_args
                and _values_match_fuzzy(actual_args[key], expected_val)
            )

            match_ratio = matched_keys / len(expected_args)

            if match_ratio == 1.0:
                matching_calls += 1
                found_exact = True
                break
            elif match_ratio > best_match_score:
                best_match_score = match_ratio

        # Track partial matches for reporting
        if not found_exact and best_match_score > 0:
            partial_matches.append((expected_name, best_match_score))

    score = matching_calls / len(expected_tool_calls)

    # Build detailed comment
    comment_parts = [
        f"Exact matches: {matching_calls}/{len(expected_tool_calls)} ({score:.1%})"
    ]
    if partial_matches:
        partial_info = ", ".join(
            f"{name}:{ratio:.0%}" for name, ratio in partial_matches[:3]
        )
        comment_parts.append(f"Partial: {partial_info}")

    return Evaluation(
        name="tool_calls_arguments",
        value=score,
        comment=" | ".join(comment_parts),
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


def create_plan_quality_evaluator(temperature: float = 0.0) -> Any:
    """Return a Langfuse-compatible plan quality evaluator function.

    The returned evaluator has the item-level signature expected by
    `run_experiment`:

        async def(*, input, output, expected_output, metadata, **kwargs)
            -> list[Evaluation]

    Parameters
    ----------
    temperature : float
        Judge model temperature. Keep at 0.0 for deterministic scoring.

    Returns
    -------
    EvaluatorFunction
        Async evaluator compatible with `run_experiment`.
    """
    rubric_path = (
        Path(__file__).parent.parent / "evaluator_prompts" / "plan_quality_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="plan_quality_judge",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
    )


def compute_composite_score_for_item(evaluations: list[Evaluation]) -> dict[str, float]:
    """Compute weighted composite score from item-level evaluations.

    Weights (totaling 1.00):
    - Plan Quality: 0.30 (strategic foundation)
    - Arguments: 0.25 (execution correctness)
    - F1: 0.20 (precision/recall balance)
    - Coverage: 0.15 (breadth of tool usage)
    - Trajectory: 0.10 (sequence correctness)

    Parameters
    ----------
    evaluations : list[Evaluation]
        List of evaluation results for a single item.

    Returns
    -------
    dict[str, float]
        Dictionary with 'composite_score' and individual metric contributions.
    """
    # Define weights
    weights = {
        "plan_quality": 0.30,
        "tool_calls_arguments": 0.25,
        "tool_calls_f1": 0.20,
        "tool_calls_coverage": 0.15,
        "tool_calls_trajectory": 0.10,
    }

    # Extract scores from evaluations
    scores = {}
    for eval_result in evaluations:
        if eval_result.name in weights:
            scores[eval_result.name] = eval_result.value

    # Calculate weighted score
    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        if metric_name in scores:
            weighted_sum += scores[metric_name] * weight
            total_weight += weight

    # Normalize by actual total weight (in case some metrics are missing)
    composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "composite_score": composite_score,
        "scores": scores,
        "total_weight": total_weight,
    }


def create_composite_evaluator_per_item():
    """Create a composite evaluator for per-item scoring.

    This is a composite evaluator that runs on each item and receives
    the evaluations from item-level evaluators for that specific item.

    Returns
    -------
    Callable
        Composite evaluator function.
    """
    from typing import Any

    def evaluate_composite_item(
        *,
        input: Any,
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any],
        evaluations: list[Evaluation],
        **kwargs: Any,
    ) -> Evaluation:
        """Composite evaluator that computes weighted score for a single item.

        Parameters
        ----------
        input : Any
            The input to the task.
        output : Any
            The output from the task.
        expected_output : Any
            The expected output.
        metadata : dict[str, Any]
            Metadata for the item.
        evaluations : list[Evaluation]
            List of evaluations from item-level evaluators for this item.

        Returns
        -------
        Evaluation
            Composite score evaluation for this item.
        """
        # Compute composite score for this item
        result = compute_composite_score_for_item(evaluations)
        composite_score = result["composite_score"]
        scores = result["scores"]
        total_weight = result["total_weight"]

        # Build comment with breakdown
        weights = {
            "plan_quality": 0.30,
            "tool_calls_arguments": 0.25,
            "tool_calls_f1": 0.20,
            "tool_calls_coverage": 0.15,
            "tool_calls_trajectory": 0.10,
        }

        comment_parts = [f"Composite: {composite_score:.3f}"]
        for metric_name, weight in weights.items():
            if metric_name in scores:
                score = scores[metric_name]
                contribution = (
                    score * weight / total_weight if total_weight > 0 else 0.0
                )
                metric_display = (
                    metric_name.replace("tool_calls_", "").replace("_", " ").title()
                )
                comment_parts.append(
                    f"{metric_display}: {score:.3f} (wgt: {weight:.2f}, contrib: {contribution:.3f})"
                )

        return Evaluation(
            name="composite_score",
            value=composite_score,
            comment=" | ".join(comment_parts),
        )

    return evaluate_composite_item


def create_composite_evaluator_run_level():
    """Create a run-level evaluator that computes overall composite score.

    This computes the average composite score across all items in the run.

    Returns
    -------
    Callable
        Run-level evaluator function.
    """
    from typing import Any

    def evaluate_composite_aggregate(
        *,
        item_results: list[Any],
        **kwargs: Any,
    ) -> Evaluation:
        """Run-level evaluator that computes overall composite score.

        Parameters
        ----------
        item_results : list[ExperimentItemResult]
            List of results from all experiment items, each containing evaluations.

        Returns
        -------
        Evaluation
            Single evaluation with average composite score across all items.
        """
        composite_scores = []

        for item_result in item_results:
            # Extract evaluations for this item
            evaluations = item_result.evaluations

            # Compute composite score for this item
            result = compute_composite_score_for_item(evaluations)
            composite_scores.append(result["composite_score"])

        # Calculate overall average
        n_items = len(composite_scores)
        avg_composite = sum(composite_scores) / n_items if n_items > 0 else 0.0

        return Evaluation(
            name="composite_score_avg",
            value=avg_composite,
            comment=f"Average composite score across {n_items} items: {avg_composite:.3f}",
        )

    return evaluate_composite_aggregate
