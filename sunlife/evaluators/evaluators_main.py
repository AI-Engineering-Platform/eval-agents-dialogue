import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.knowledge_qa import DeepSearchQADataset, KnowledgeGroundedAgent
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse

from answer_relvance import create_answer_relevance_evaluator
from correctness_evaluator import create_correctness_evaluator
from hallucination_evaluator import create_hallucination_evaluator


BASE_DIR = Path("/home/coder/eval-agents")
os.chdir(BASE_DIR)

load_dotenv()
console = Console(width=100)

DATASET_NAME = "DeepSearchQA-Sun-Life_with_tools"

async def agent_task(*, item: Any, **kwargs: Any) -> str:
    agent = KnowledgeGroundedAgent(enable_planning=True)
    response = await agent.answer_async(item.input)
    return response.text

async def main():
    dataset = DeepSearchQADataset()
    examples = dataset.get_by_category("Finance & Economics")[:10]

    console.print(f"Uploading [cyan]{len(examples)}[/cyan] examples...")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as f:
        for ex in examples:
            record = {
                "input": ex.problem,
                "expected_output": ex.answer,
                "metadata": {
                    "example_id": ex.example_id,
                    "category": ex.problem_category,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        temp_path = f.name

    await upload_dataset_to_langfuse(
        dataset_path=temp_path,
        dataset_name=DATASET_NAME,
    )

    os.unlink(temp_path)

    console.print("[green]✓ Dataset uploaded[/green]")

    evaluators = [
        # create_toxicity_evaluator(),
        create_answer_relevance_evaluator(),
        create_correctness_evaluator(),
        create_hallucination_evaluator(),
    ]

    console.print("\n[bold yellow]🚀 Running Experiment...[/bold yellow]\n")

    # NOTE: run_experiment is synchronous
    experiment_result = run_experiment(
        DATASET_NAME,
        name="multiple2-eval-answer-correct",
        task=agent_task,
        evaluators=evaluators,
        max_concurrency=1,
    )

    console.print("\n[bold green]✓ Experiment completed successfully[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())