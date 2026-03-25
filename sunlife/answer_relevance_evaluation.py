"""Answer relevance evaluation pipeline for the Knowledge QA agent."""
import asyncio
import logging
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
from dotenv import load_dotenv
from langfuse.experiment import Evaluation
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv(verbose=True)
logger = logging.getLogger(__name__)
console = Console(width=100)

ANSWER_RELEVANCE_RUBRIC = """
SYSTEM ROLE:
You are an expert evaluator assessing whether a retrieval-based AI agent provides a relevant and complete answer to a user question.

TASK:
Evaluate ONLY the Candidate Output in relation to the Input Question.

SCORING RULES:
- Use binary scoring (0 or 1) for each metric.
- Provide a one-sentence justification for each score.
- Focus on relevance, completeness, and alignment with the user's intent.

METRICS:

1. is_relevant
   - 1 if the response directly addresses the core intent of the question
   - 0 if the response is off-topic, partially relevant, or unrelated

2. completeness
   - 1 if the response fully answers ALL components of the question
   - 0 if any part of the question is missing or insufficiently addressed

3. intent_alignment
   - 1 if the response matches the expected format and intent (e.g., list, fact, explanation)
   - 0 if the response deviates from the expected structure or intent

4. no_extraneous_information
   - 1 if the response contains ONLY information required to answer the question
   - 0 if it includes unnecessary explanations, tangents, or unrelated content

EVALUATION GUIDELINES:
- Ignore tone, style, or verbosity unless it affects relevance.
- For "Set Answer" questions, ensure ALL required elements are covered.
- For "Single Answer" questions, ensure a direct and precise response.
"""

def create_answer_relevance_evaluator(temperature: float = 0.0) -> Any:
    return create_llm_as_judge_evaluator(
        name="answer_relevance_judge",
        rubric_markdown=ANSWER_RELEVANCE_RUBRIC,
        model_config=LLMRequestConfig(temperature=temperature),
    )

async def run_local(question: str) -> list[Evaluation]:
    console.print("[dim]Running agent...[/dim]\n")
    agent = KnowledgeGroundedAgent(enable_planning=True)
    response = await agent.answer_async(question)

    console.print(Panel(response.text[:500], title="Agent Answer", border_style="blue"))

    evaluator = create_answer_relevance_evaluator()
    evaluations = await evaluator(
        input=question,
        output=response.text,
        expected_output="",
        metadata=None,
    )

    _display_results(evaluations)
    return evaluations

def _display_results(evaluations: list[Evaluation]) -> None:
    table = Table(title="Answer Relevance Results")
    table.add_column("Metric")
    table.add_column("Score")
    table.add_column("Comment")

    for ev in evaluations:
        table.add_row(ev.name, str(ev.value), ev.comment or "")

    console.print(table)

async def _agent_task(*, item: Any, **kwargs: Any) -> str:
    agent = KnowledgeGroundedAgent(enable_planning=True)
    response = await agent.answer_async(item.input)
    return response.text

def run_experiment_eval(dataset_name: str, experiment_name: str):
    evaluator = create_answer_relevance_evaluator()

    run_experiment(
        dataset_name,
        name=experiment_name,
        task=_agent_task,
        evaluators=[evaluator],
    )

@click.command()
@click.option("--local", is_flag=True)
@click.option("--question", default="What is the Federal Reserve?")
@click.option("--dataset-name", default="DeepSearchQA-Sun-Life_with_tools")
@click.option("--experiment-name", default="answer-relevance-eval")
def cli(local, question, dataset_name, experiment_name):
    if local:
        asyncio.run(run_local(question))
    else:
        run_experiment_eval(dataset_name, experiment_name)

if __name__ == "__main__":
    cli()