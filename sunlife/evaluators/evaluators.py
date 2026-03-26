"""
Evaluators for tool calls, plan quality, and source reliability.

Includes:
- Deterministic evaluators: coverage, F1, arguments, trajectory
- LLM-as-judge evaluators: plan quality, source reliability
- Composite evaluator: weighted combination of all metrics
"""

from pathlib import Path
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from langfuse.experiment import Evaluation
from pydantic import BaseModel


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


def evaluate_coverage(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """
    Evaluate tool call coverage.
    Checks if all expected tool types were called at least once.
    Ignores number of calls and arguments.

    Score: Percentage of expected tool types that were called (0.0 to 1.0)
    """
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])

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


def evaluate_f1(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """
    Evaluate tool call F1 score.
    Evaluates tool calls using precision and recall, computing F1 score.
    Compares each expected tool call against actual tool calls (1:1 matching).

    Score: F1 score (0.0 to 1.0)
    """
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])

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


def evaluate_arguments(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """
    Evaluate tool call arguments.
    Evaluates if tool arguments match using fuzzy comparison.
    Compares each expected tool call's arguments with actual tool calls.
    Uses normalized comparison with partial matching for strings.

    Score: Percentage of expected tool calls with matching arguments (0.0 to 1.0)
    """
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])

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


def evaluate_trajectory(
    *,
    output: dict[str, Any],
    metadata: dict[str, Any],
    **kwargs: Any,
) -> Evaluation:
    """
    Evaluate tool call sequence/trajectory.
    Evaluates the sequence/order of tool calls using edit distance.
    Measures how similar the actual tool call sequence is to the expected sequence.

    Score: Normalized sequence similarity (0.0 to 1.0)
           1.0 = perfect match, 0.0 = completely different
    """
    actual_tool_calls = output.get("tool_calls", [])
    expected_tool_calls = metadata.get("expected_tool_calls", [])

    if not expected_tool_calls:
        score = 1.0 if not actual_tool_calls else 0.0
        return Evaluation(
            name="tool_calls_trajectory",
            value=score,
            comment=(
                "No expected tool calls"
                if not actual_tool_calls
                else "Used tools when none expected"
            ),
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


def create_toxicity_evaluator(temperature: float = 0.0) -> Any:
    """Return a Langfuse-compatible toxicity evaluator function.

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
        Path(__file__).parent.parent / "evaluator_prompts" / "toxicity_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="toxicity_judge",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
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


def create_source_reliability_evaluator(temperature: float = 0.0):
    """Create an LLM-as-judge evaluator for source reliability.

    Evaluates whether the sources used to answer a question are reliable,
    relevant, credible, sufficient, and appropriate - similar to evaluating
    internet sources for academic research.

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
        Path(__file__).parent.parent
        / "evaluator_prompts"
        / "source_reliability_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="source_reliability",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
    )


# Exact duplicate (tool_name, args) — rule-based
async def redundancy_tool_call_evaluator(
    *,
    input: str,
    output: Any,
    expected_output: Any,
    metadata: Any = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate exact duplicate tool calls: same tool name and same args."""
    tool_calls = output.get("tool_calls", []) if isinstance(output, dict) else []

    if not tool_calls:
        return [
            Evaluation(
                name="redundancy_ratio", value=0.0, comment="No tool calls found"
            )
        ]

    seen = []
    duplicates = 0
    for tc in tool_calls:
        key = (tc.get("name"), str(tc.get("args", {})))
        if key in seen:
            duplicates += 1
        else:
            seen.append(key)

    redundancy_ratio = duplicates / len(tool_calls)

    return [
        Evaluation(
            name="redundancy_ratio",
            value=round(redundancy_ratio, 3),
            comment=f"{duplicates} exact duplicate calls out of {len(tool_calls)} total",
        )
    ]


# Duplicate URLs (web_fetch) — rule-based
async def duplicate_url_evaluator(
    *,
    input: str,
    output: Any,
    expected_output: Any,
    metadata: Any = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate whether the agent fetched the same URL more than once."""
    tool_calls = output.get("tool_calls", []) if isinstance(output, dict) else []

    fetch_calls = [
        tc for tc in tool_calls if tc.get("name") in ("web_fetch", "fetch_file")
    ]

    if not fetch_calls:
        return [
            Evaluation(
                name="duplicate_url_ratio", value=0.0, comment="No fetch calls found"
            )
        ]

    urls = [str(tc.get("args", {}).get("url", "")) for tc in fetch_calls]
    seen_urls: list[str] = []
    duplicate_urls: list[str] = []

    for url in urls:
        if url and url in seen_urls:
            duplicate_urls.append(url)
        else:
            seen_urls.append(url)

    duplicate_url_ratio = len(duplicate_urls) / len(fetch_calls)

    return [
        Evaluation(
            name="duplicate_url_ratio",
            value=round(duplicate_url_ratio, 3),
            comment=(
                f"{len(duplicate_urls)} duplicate URLs out of {len(fetch_calls)} fetch calls"
                if duplicate_urls
                else f"No duplicate URLs across {len(fetch_calls)} fetch calls"
            ),
        )
    ]


SEMANTIC_REDUNDANCY_SYSTEM_PROMPT = """\
You are evaluating whether a list of search queries issued by an AI agent are semantically redundant.

Two queries are semantically redundant if they would retrieve substantially overlapping information,
even if they use different words, abbreviations, or phrasing.

Examples of redundant pairs:
- "Federal Reserve interest rate hike" and "Fed rate increase 2024"
- "causes of 2008 financial crisis" and "what caused the subprime mortgage collapse"

Examples of non-redundant pairs:
- "Basel III capital requirements" and "Basel III implementation timeline"
- "Federal Reserve mandate" and "Federal Reserve interest rate history"

Return valid JSON only. Do not use Markdown code blocks.
Schema:
{
  "redundant_pairs": [["query A", "query B"], ...],
  "reasoning": "brief explanation"
}
"""


class SemanticRedundancyResponse(BaseModel):
    redundant_pairs: list[list[str]]
    reasoning: str


# Semantic query overlap (google_search) — LLM-as-judge
async def semantic_query_redundancy_evaluator(
    *,
    input: str,
    output: Any,
    expected_output: Any,
    metadata: Any = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate whether the agent issued semantically redundant search queries."""
    tool_calls = output.get("tool_calls", []) if isinstance(output, dict) else []

    search_calls = [
        tc
        for tc in tool_calls
        if tc.get("name") in ("google_search", "google_search_agent")
    ]

    if len(search_calls) < 2:
        return [
            Evaluation(
                name="semantic_query_redundancy",
                value=0.0,
                comment="Fewer than 2 search queries — nothing to compare",
            )
        ]

    queries = [
        str(tc.get("args", {}).get("query", tc.get("args", ""))) for tc in search_calls
    ]
    queries_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))

    user_prompt = f"Here are the search queries issued by the agent:\n\n{queries_text}\n\nIdentify any semantically redundant pairs."

    try:
        client_manager = AsyncClientManager.get_instance()
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=LLMRequestConfig(temperature=0.0),
            system_prompt=SEMANTIC_REDUNDANCY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=SemanticRedundancyResponse,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("LLM returned no structured output")

        total_pairs = len(queries) * (len(queries) - 1) // 2  # n choose 2
        redundant_count = len(result.redundant_pairs)
        redundancy_score = redundant_count / total_pairs if total_pairs > 0 else 0.0

        comment = result.reasoning
        if result.redundant_pairs:
            pair_strs = "; ".join(f'"{a}" ~ "{b}"' for a, b in result.redundant_pairs)
            comment = f"Redundant pairs: {pair_strs}. {result.reasoning}"

        return [
            Evaluation(
                name="semantic_query_redundancy",
                value=round(redundancy_score, 3),
                comment=comment,
            )
        ]

    except Exception as e:
        return [
            Evaluation(
                name="semantic_query_redundancy",
                value=0.0,
                comment=f"Evaluation error: {e}",
            )
        ]


def evaluate_composite(
    *,
    evaluations: list[Evaluation],
    **kwargs: Any,
) -> Evaluation:
    """Composite evaluator that computes weighted score for a single item.

    Weights (totaling 1.00):
    - Correctness: 0.17 (factual accuracy and grounding)
    - Answer Relevance: 0.15 (relevance and completeness)
    - Hallucination: 0.13 (inverted hallucination = grounding quality)
    - Plan Quality: 0.12 (strategic foundation)
    - Source Reliability: 0.10 (source credibility)
    - Answer Clarity: 0.10 (understandability and structure)
    - Arguments: 0.09 (execution correctness)
    - F1: 0.07 (precision/recall balance)
    - Coverage: 0.03 (breadth of tool usage)
    - Trajectory: 0.02 (sequence correctness)
    - Toxicity: 0.01 (inverted toxicity = safety)
    - Redundancy Ratio: 0.01 (duplicate tool calls - inverted)
    - Duplicate URL Ratio: 0.005 (duplicate fetches - inverted)
    - Semantic Query Redundancy: 0.005 (redundant searches - inverted)

    Note: Redundancy/toxicity metrics are inverted (1 - value) so lower = higher score.
    """
    weights = {
        "correctness": 0.17,
        "answer_relevance": 0.15,
        "hallucination": 0.13,
        "plan_quality": 0.12,
        "source_reliability": 0.10,
        "answer_clarity": 0.10,
        "tool_calls_arguments": 0.09,
        "tool_calls_f1": 0.07,
        "tool_calls_coverage": 0.03,
        "tool_calls_trajectory": 0.02,
        "toxicity_score": 0.01,  # Inverted: lower is better
        "redundancy_ratio": 0.01,  # Inverted: lower is better
        "duplicate_url_ratio": 0.005,  # Inverted: lower is better
        "semantic_query_redundancy": 0.005,  # Inverted: lower is better
    }

    # Extract scores from evaluations
    # Redundancy metrics need to be inverted (lower redundancy = better score)
    redundancy_metrics = {
        "redundancy_ratio",
        "duplicate_url_ratio",
        "semantic_query_redundancy",
    }

    scores = {}
    for eval_result in evaluations:
        if eval_result.name in weights:
            value = eval_result.value
            # Invert redundancy metrics: 1.0 - value (so 0 redundancy = 1.0 score)
            if eval_result.name in redundancy_metrics:
                value = 1.0 - value
            scores[eval_result.name] = value

    # Calculate weighted score
    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in weights.items():
        if metric_name in scores:
            weighted_sum += scores[metric_name] * weight
            total_weight += weight

    composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Build comment
    comment_parts = [f"Composite: {composite_score:.3f}"]
    for metric_name, weight in weights.items():
        if metric_name in scores:
            score = scores[metric_name]
            contribution = score * weight / total_weight if total_weight > 0 else 0.0
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


def create_answer_relevance_evaluator(temperature: float = 0.0) -> Any:
    """Return a Langfuse-compatible answer relevance evaluator function.

    Evaluates whether the agent's answer is relevant, complete, and aligned
    with the user's intent.

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
        Path(__file__).parent.parent
        / "evaluator_prompts"
        / "answer_relevance_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="answer_relevance_judge",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
    )


def create_correctness_evaluator(temperature: float = 0.0) -> Any:
    """Return a Langfuse-compatible correctness evaluator function.

    Evaluates factual correctness and reasoning quality of the agent response.

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
        Path(__file__).parent.parent / "evaluator_prompts" / "correctness_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="correctness_judge",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
    )


def create_hallucination_evaluator(temperature: float = 0.0) -> Any:
    """Return a Langfuse-compatible hallucination evaluator function.

    Evaluates whether the response is grounded and free from hallucinations.

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
        Path(__file__).parent.parent / "evaluator_prompts" / "hallucination_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="hallucination_judge",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
    )


def create_answer_clarity_evaluator(temperature: float = 0.0):
    """Create an LLM-as-judge evaluator for answer clarity

    Evaluates answer's understandability, conciseness, and structure.

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
        Path(__file__).parent.parent / "evaluator_prompts" / "answer_clarity_rubric.txt"
    )

    return create_llm_as_judge_evaluator(
        name="answer_clarity",
        rubric_markdown=rubric_path,
        model_config=LLMRequestConfig(temperature=temperature),
    )
