"""
Run tool call evaluation experiment on the enriched dataset from Langfuse.
"""

from typing import Any

from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.knowledge_qa import KnowledgeGroundedAgent
from dotenv import load_dotenv
from evaluators.langfuse_evaluators import (
    evaluate_arguments,
    evaluate_composite,
    evaluate_coverage,
    evaluate_f1,
    evaluate_trajectory,
)
from evaluators.evaluators import create_plan_quality_evaluator
from rich.console import Console
from rich.table import Table

load_dotenv(verbose=True)
console = Console(width=120)

DATASET_NAME = "DeepSearchQA-Sun-Life_with_tools"


async def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the agent on the input question and return tool calls."""
    agent = KnowledgeGroundedAgent()
    response = await agent.answer_async(item.input)
    agent.reset()

    return {
        "answer": response.text,
        "tool_calls": response.tool_calls,
        "sources": [{"title": s.title, "uri": s.uri} for s in response.sources],
        "plan": {
            "reasoning": response.plan.reasoning,
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "step_type": step.step_type,
                    "expected_output": step.expected_output,
                    "status": step.status,
                }
                for step in response.plan.steps
            ],
        },
    }


def main():
    """Run the tool call evaluation experiment."""
    console.print(
        f"[cyan]Running tool call evaluation experiment on dataset: {DATASET_NAME}[/cyan]"
    )

    experiment_result = run_experiment(
        DATASET_NAME,
        name="knowledge-agent-tool-calls",
        task=task,
        evaluators=[
            evaluate_coverage,
            evaluate_f1,
            evaluate_arguments,
            evaluate_trajectory,
            create_plan_quality_evaluator(),
            evaluate_composite,
        ],
        description="Evaluate tool call accuracy and plan quality - coverage, F1, arguments, trajectory, tool correctness judge, and plan quality judge.",
        max_concurrency=1,
    )

    console.print(f"\n[green]✓[/green] Experiment completed!")
    console.print(f"[cyan]Results published to Langfuse[/cyan]")

    # Print summary table
    if experiment_result.item_results:
        table = Table(title="Tool Call Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Avg Score", justify="right")
        table.add_column("Count", justify="right")

        # Aggregate scores by metric name
        metrics = {}
        for item in experiment_result.item_results:
            for eval_result in item.evaluations:
                metric_name = eval_result.name
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(eval_result.value)

        for metric_name, scores in metrics.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            table.add_row(
                metric_name.replace("tool_calls_", "").replace("_", " ").title(),
                f"{avg_score:.2f}",
                str(len(scores)),
            )

        console.print("\n")
        console.print(table)


if __name__ == "__main__":
    main()
