"""Hallucination evaluation pipeline for the Knowledge QA agent."""
import asyncio
import logging
from typing import Any

import click
from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
from dotenv import load_dotenv
from langfuse.experiment import Evaluation
from rich.console import Console

load_dotenv(verbose=True)
logger = logging.getLogger(__name__)
console = Console(width=100)

HALLUCINATION_RUBRIC = """
SYSTEM ROLE:
You are an expert evaluator of retrieval-grounded AI systems.

TASK:
Determine whether the response is fully grounded in provided evidence (tool outputs, citations, or widely known facts).

SCORING RULES:
- Use binary scoring (0 or 1) per metric.
- Be strict: any unsupported factual claim → 0.

METRICS:

1. is_grounded
   - 1 if every factual claim is supported by retrieved evidence or universally accepted knowledge
   - 0 if ANY unsupported, fabricated, or unverifiable claim exists

2. no_hallucinated_entities
   - 1 if all people, places, numbers, sources, and claims are real and verifiable
   - 0 if any invented entity, number, or source appears

3. citation_validity
   - 1 if all cited sources actually support the claims they are attached to
   - 0 if citations are fake, misused, or irrelevant

4. uncertainty_handling
   - 1 if uncertainty is expressed when evidence is insufficient
   - 0 if the model presents uncertain or speculative claims as facts

GUIDELINES:
- A single hallucinated claim → is_grounded = 0
- Fabricated citations are ALWAYS 0
- Overclaiming without evidence is hallucination
"""

def create_hallucination_evaluator(temperature: float = 0.0) -> Any:
    return create_llm_as_judge_evaluator(
        name="hallucination_judge",
        rubric_markdown=HALLUCINATION_RUBRIC,
        model_config=LLMRequestConfig(temperature=temperature),
    )

async def run_local(question: str) -> list[Evaluation]:
    agent = KnowledgeGroundedAgent(enable_planning=True)
    response = await agent.answer_async(question)

    evaluator = create_hallucination_evaluator()
    return await evaluator(input=question, output=response.text, expected_output="", metadata=None)

async def _agent_task(*, item: Any, **kwargs: Any) -> str:
    agent = KnowledgeGroundedAgent(enable_planning=True)
    response = await agent.answer_async(item.input)
    return response.text

def run_experiment_eval(dataset_name: str, experiment_name: str):
    evaluator = create_hallucination_evaluator()

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
@click.option("--experiment-name", default="hallucination-eval")
def cli(local, question, dataset_name, experiment_name):
    if local:
        asyncio.run(run_local(question))
    else:
        run_experiment_eval(dataset_name, experiment_name)

if __name__ == "__main__":
    cli()