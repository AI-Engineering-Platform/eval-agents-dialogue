"""
Run tool call evaluation experiment on the enriched dataset from Langfuse.
"""

from typing import Any

from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.knowledge_qa import KnowledgeGroundedAgent
from dotenv import load_dotenv
from evaluators.evaluators import (create_composite_evaluator_per_item,
                                   create_plan_quality_evaluator,
                                   create_source_reliability_evaluator,
                                   evaluate_arguments, evaluate_coverage,
                                   evaluate_f1, evaluate_trajectory,
                                   redundancy_tool_call_evaluator, 
                                   duplicate_url_evaluator, 
                                   semantic_query_redundancy_evaluator)
from rich.console import Console

load_dotenv(verbose=True)
console = Console(width=120)

DATASET_NAME = "DeepSearchQA-Sun-Life-Tool-calls-2"


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
            create_source_reliability_evaluator(),
            redundancy_tool_call_evaluator,
            duplicate_url_evaluator,
            semantic_query_redundancy_evaluator
        ],
        composite_evaluator=create_composite_evaluator_per_item(),
        description="Evaluate tool call accuracy and plan quality - coverage, F1, arguments, trajectory, tool correctness judge, plan quality judge, and composite score (per-item and aggregate).",
        max_concurrency=1,
    )

    console.print(f"\n[green]✓[/green] Experiment completed!")
    console.print(f"[cyan]Results published to Langfuse[/cyan]")

if __name__ == "__main__":
    main()
