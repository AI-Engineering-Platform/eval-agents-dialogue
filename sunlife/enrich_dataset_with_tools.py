import asyncio
import json
import os
import tempfile
from pathlib import Path

from aieng.agent_evals.knowledge_qa import DeepSearchQADataset, KnowledgeGroundedAgent
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from dotenv import load_dotenv
from rich.console import Console

load_dotenv(verbose=True)
console = Console(width=100)

DATASET_NAME = "DeepSearchQA-Sun-Life-Tool-calls-2"


async def main():
    # Load the dataset examples
    dataset = DeepSearchQADataset()
    examples = (
        dataset.get_by_category("Finance & Economics")[:1]
        + dataset.get_by_category("Politics & Government")[:1]
        + dataset.get_by_category("Health")[:1]
    )

    console.print(
        f"Loaded [cyan]{len(examples)}[/cyan] examples from multiple categories"
    )

    # Step 1: Run agent on all examples and collect tool calls
    console.print("[cyan]Running agent on examples to collect tool calls...[/cyan]")

    agent = KnowledgeGroundedAgent()
    enriched_examples = []

    for i, ex in enumerate(examples):
        console.print(
            f"[yellow]Processing example {i+1}/{len(examples)}[/yellow]: {ex.problem[:80]}..."
        )

        # Run agent and get response (use async version)
        response = await agent.answer_async(ex.problem)

        # Add tool calls to example
        ex.expected_tool_calls = response.tool_calls
        enriched_examples.append(ex)

        # Reset agent state for next question
        agent.reset()

        console.print(
            f"  [green][/green] Collected {len(response.tool_calls)} tool calls"
        )

    console.print(f"[green][/green] Finished collecting tool calls for all examples")

    # Step 2: Save enriched dataset to file
    output_dir = Path("sunlife")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "enriched_dataset.jsonl"

    console.print(f"[cyan]Saving enriched dataset to {output_path}...[/cyan]")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in enriched_examples:
            record = {
                "example_id": ex.example_id,
                "problem": ex.problem,
                "problem_category": ex.problem_category,
                "answer": ex.answer,
                "answer_type": ex.answer_type,
                "expected_tool_calls": ex.expected_tool_calls,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(
        f"[green][/green] Saved enriched dataset with {len(enriched_examples)} examples"
    )

    # Step 3: Upload enriched dataset to Langfuse
    console.print(f"[cyan]Uploading enriched dataset to Langfuse...[/cyan]")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for ex in enriched_examples:
            record = {
                "input": ex.problem,
                "expected_output": ex.answer,
                "metadata": {
                    "example_id": ex.example_id,
                    "category": ex.problem_category,
                    "answer_type": ex.answer_type,
                    "expected_tool_calls": ex.expected_tool_calls,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path = f.name

    await upload_dataset_to_langfuse(dataset_path=temp_path, dataset_name=DATASET_NAME)
    os.unlink(temp_path)

    console.print(
        f"[green][/green] Dataset '{DATASET_NAME}' uploaded to Langfuse with tool calls"
    )


if __name__ == "__main__":
    asyncio.run(main())
