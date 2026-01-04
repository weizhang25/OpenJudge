# -*- coding: utf-8 -*-
"""Rubric generator for zero-shot evaluation."""

from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from cookbooks.zero_shot_evaluation.core.schema import OpenAIEndpoint, TaskConfig
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import PromptTemplate

RUBRIC_GENERATION_PROMPT = """# Task
Generate evaluation rubrics for pairwise comparison of model responses.

## Task Description
{task_description}

## Scenario
{scenario}

## Sample Queries (for context)
{sample_queries}

## Requirements
- Generate 3-5 clear evaluation criteria for comparing two responses
- Each criterion should be objective and measurable
- Criteria should be relevant to the task and scenario
- Focus on aspects that distinguish good responses from poor ones

## Output Format
Return a JSON object with:
- rubrics: list of evaluation criteria strings
- reason: brief explanation of why these criteria are important

Example:
{{
    "rubrics": [
        "Accuracy: Whether the response contains correct and factual information",
        "Completeness: Whether the response fully addresses the query",
        "Clarity: Whether the response is well-organized and easy to understand"
    ],
    "reason": "These criteria capture the key aspects for evaluating..."
}}
"""

RUBRIC_GENERATION_TEMPLATE = PromptTemplate(
    messages=[
        ChatMessage(
            role="system",
            content="You are an expert at designing evaluation criteria for AI systems.",
        ),
        ChatMessage(role="user", content=RUBRIC_GENERATION_PROMPT),
    ],
)


class RubricGenerationOutput(BaseModel):
    """Output schema for rubric generation."""

    rubrics: List[str] = Field(..., description="List of evaluation rubrics")
    reason: str = Field(default="", description="Reasoning for these rubrics")


class RubricGenerator:
    """Generate evaluation rubrics based on task description."""

    def __init__(
        self,
        judge_endpoint: OpenAIEndpoint,
        task_config: TaskConfig,
    ):
        """Initialize RubricGenerator.

        Args:
            judge_endpoint: OpenAI-compatible endpoint for generation
            task_config: Task configuration
        """
        self.task_config = task_config

        extra_params = judge_endpoint.extra_params or {}
        self.model = OpenAIChatModel(
            model=judge_endpoint.model,
            api_key=judge_endpoint.api_key,
            base_url=judge_endpoint.base_url,
            **extra_params,
        )

    async def generate(
        self,
        sample_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate evaluation rubrics.

        Args:
            sample_queries: Optional sample queries for context

        Returns:
            List of rubric strings
        """

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.0))
        async def _generate() -> List[str]:
            queries_text = "None provided"
            if sample_queries:
                queries_text = "\n".join(f"- {q}" for q in sample_queries[:5])

            messages = RUBRIC_GENERATION_TEMPLATE.format(
                task_description=self.task_config.description,
                scenario=self.task_config.scenario or "General usage",
                sample_queries=queries_text,
            )

            response = await self.model.achat(
                messages=list(messages),
                structured_model=RubricGenerationOutput,
            )

            if not response.parsed or "rubrics" not in response.parsed:
                raise ValueError("Failed to parse rubric generation response")

            return response.parsed["rubrics"]

        try:
            rubrics = await _generate()
            logger.info(f"Generated {len(rubrics)} evaluation rubrics")
            for i, rubric in enumerate(rubrics, 1):
                logger.debug(f"  {i}. {rubric}")
            return rubrics
        except Exception as e:
            logger.error(f"Rubric generation failed: {e}")
            # Return default rubrics as fallback
            return [
                "Accuracy: Whether the response is factually correct",
                "Relevance: Whether the response addresses the query",
                "Completeness: Whether the response is comprehensive",
            ]

