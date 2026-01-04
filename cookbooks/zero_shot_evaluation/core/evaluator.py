# -*- coding: utf-8 -*-
"""End-to-end evaluator for zero-shot evaluation."""

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field

from cookbooks.zero_shot_evaluation.core.checkpoint import (
    CheckpointManager,
    EvaluationStage,
)
from cookbooks.zero_shot_evaluation.core.config import load_config
from cookbooks.zero_shot_evaluation.core.query_generator import QueryGenerator
from cookbooks.zero_shot_evaluation.core.response_collector import ResponseCollector
from cookbooks.zero_shot_evaluation.core.rubric_generator import RubricGenerator
from cookbooks.zero_shot_evaluation.core.schema import (
    GeneratedQuery,
    OpenAIEndpoint,
    ZeroShotConfig,
)
from openjudge.graders.llm_grader import GraderMode, LLMGrader
from openjudge.graders.schema import GraderResult, GraderScore
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import PromptTemplate
from openjudge.runner.grading_runner import GraderConfig, GradingRunner


class EvaluationResult(BaseModel):
    """Result of zero-shot evaluation."""

    rankings: List[Tuple[str, float]] = Field(default_factory=list)
    win_rates: Dict[str, float] = Field(default_factory=dict)
    win_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    best_pipeline: str = Field(default="")
    total_queries: int = Field(default=0)
    total_comparisons: int = Field(default=0)


class ZeroShotEvaluator:
    """End-to-end zero-shot evaluator with checkpoint support."""

    def __init__(
        self,
        config: Optional[ZeroShotConfig] = None,
        *,
        task_description: Optional[str] = None,
        target_endpoints: Optional[Dict[str, OpenAIEndpoint]] = None,
        judge_endpoint: Optional[OpenAIEndpoint] = None,
        num_queries: int = 20,
        resume: bool = True,
    ):
        """Initialize ZeroShotEvaluator.

        Args:
            config: Complete configuration object
            task_description: Task description (alternative to config)
            target_endpoints: Target endpoints (alternative to config)
            judge_endpoint: Judge endpoint (alternative to config)
            num_queries: Number of queries to generate
            resume: Whether to resume from checkpoint if available
        """
        if config:
            self.config = config
        else:
            if not all([task_description, target_endpoints, judge_endpoint]):
                raise ValueError("Must provide either config or all individual parameters")
            from cookbooks.zero_shot_evaluation.core.schema import (
                EvaluationConfig,
                QueryGenerationConfig,
                TaskConfig,
            )

            self.config = ZeroShotConfig(
                task=TaskConfig(description=task_description),
                target_endpoints=target_endpoints,
                judge_endpoint=judge_endpoint,
                query_generation=QueryGenerationConfig(num_queries=num_queries),
                evaluation=EvaluationConfig(),
            )

        self._queries: List[GeneratedQuery] = []
        self._responses: List[Dict[str, Any]] = []
        self._rubrics: List[str] = []
        
        # Initialize checkpoint manager
        self._checkpoint_mgr = CheckpointManager(self.config.output.output_dir)
        self._resume = resume

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "ZeroShotEvaluator":
        """Create evaluator from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ZeroShotEvaluator instance
        """
        config = load_config(config_path)
        return cls(config=config)

    async def generate_queries(self) -> List[GeneratedQuery]:
        """Step 1: Generate test queries."""
        logger.info("Step 1: Generating test queries...")
        generator = QueryGenerator(
            judge_endpoint=self.config.judge_endpoint,
            task_config=self.config.task,
            query_config=self.config.query_generation,
        )
        self._queries = await generator.generate()
        return self._queries

    async def collect_responses(
        self,
        queries: Optional[List[GeneratedQuery]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2: Collect responses from all target endpoints."""
        queries = queries or self._queries
        if not queries:
            raise ValueError("No queries available. Run generate_queries() first.")

        logger.info("Step 2: Collecting responses from target endpoints...")
        collector = ResponseCollector(
            target_endpoints=self.config.target_endpoints,
            evaluation_config=self.config.evaluation,
        )
        self._responses = await collector.collect(queries)
        return self._responses

    async def generate_rubrics(
        self,
        sample_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """Step 3: Generate evaluation rubrics."""
        logger.info("Step 3: Generating evaluation rubrics...")

        if not sample_queries and self._queries:
            sample_queries = [q.query for q in self._queries[:5]]

        generator = RubricGenerator(
            judge_endpoint=self.config.judge_endpoint,
            task_config=self.config.task,
        )
        self._rubrics = await generator.generate(sample_queries)
        return self._rubrics

    def _prepare_pairwise_data(
        self,
        responses: List[Dict[str, Any]],
    ) -> Tuple[List[dict], List[str]]:
        """Prepare pairwise comparison dataset.

        Args:
            responses: List of response data from collect_responses()

        Returns:
            Tuple of (dataset, endpoint_names)
        """
        endpoint_names = list(self.config.target_endpoints.keys())
        pairs = list(combinations(endpoint_names, 2))

        dataset = []
        for resp_data in responses:
            query = resp_data["query"]
            resp_dict = resp_data["responses"]

            for ep_a, ep_b in pairs:
                resp_a = resp_dict.get(ep_a)
                resp_b = resp_dict.get(ep_b)

                if resp_a is None or resp_b is None:
                    continue

                # Original order
                dataset.append(
                    {
                        "evaluation_data": {
                            "instruction": query,
                            "response_a": resp_a,
                            "response_b": resp_b,
                        },
                        "metadata": {
                            "model_a": ep_a,
                            "model_b": ep_b,
                            "order": "original",
                        },
                    }
                )
                # Swapped order (to eliminate position bias)
                dataset.append(
                    {
                        "evaluation_data": {
                            "instruction": query,
                            "response_a": resp_b,
                            "response_b": resp_a,
                        },
                        "metadata": {
                            "model_a": ep_b,
                            "model_b": ep_a,
                            "order": "swapped",
                        },
                    }
                )

        return dataset, endpoint_names

    def _build_pairwise_grader(self, rubrics: List[str]) -> LLMGrader:
        """Build pairwise comparison grader."""
        rubrics_text = "\n".join(f"- {r}" for r in rubrics)

        template = PromptTemplate(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are an expert evaluator. Compare two responses based on the given criteria.\n"
                    f"Evaluation Criteria:\n{rubrics_text}\n\n"
                    "Output JSON with 'score' (1.0 if Response A is better, 0.0 if Response B is better) "
                    "and 'reason' (brief explanation).",
                ),
                ChatMessage(
                    role="user",
                    content="Query: {instruction}\n\n"
                    "Response A:\n{response_a}\n\n"
                    "Response B:\n{response_b}\n\n"
                    "Which response is better based on the criteria?",
                ),
            ],
        )

        endpoint = self.config.judge_endpoint
        extra_params = endpoint.extra_params or {}

        return LLMGrader(
            name="pairwise_comparator",
            mode=GraderMode.POINTWISE,
            model=OpenAIChatModel(
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                temperature=extra_params.get("temperature", 0.1),
            ),
            template=template,
        )

    async def _run_pairwise_evaluation(
        self,
        dataset: List[dict],
        rubrics: List[str],
    ) -> List[GraderResult]:
        """Run pairwise evaluation using GradingRunner."""
        grader = self._build_pairwise_grader(rubrics)

        mapper = {
            "instruction": "evaluation_data.instruction",
            "response_a": "evaluation_data.response_a",
            "response_b": "evaluation_data.response_b",
        }

        runner = GradingRunner(
            grader_configs={
                "pairwise": GraderConfig(grader=grader, mapper=mapper),
            },
            max_concurrency=self.config.evaluation.max_concurrency,
        )

        logger.info(f"Running {len(dataset)} pairwise comparisons...")
        results = await runner.arun(dataset)
        return results["pairwise"]

    def _analyze_results(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        endpoint_names: List[str],
    ) -> EvaluationResult:
        """Analyze pairwise comparison results."""
        # Initialize counters
        win_counts: Dict[str, Dict[str, int]] = {
            ep: {other: 0 for other in endpoint_names if other != ep} for ep in endpoint_names
        }
        comparison_counts: Dict[str, Dict[str, int]] = {
            ep: {other: 0 for other in endpoint_names if other != ep} for ep in endpoint_names
        }

        # Count wins
        for sample, result in zip(dataset, grader_results):
            metadata = sample.get("metadata", {})
            model_a = metadata.get("model_a")
            model_b = metadata.get("model_b")

            if not model_a or not model_b or not isinstance(result, GraderScore):
                continue

            if result.score >= 0.5:
                win_counts[model_a][model_b] += 1
            else:
                win_counts[model_b][model_a] += 1

            comparison_counts[model_a][model_b] += 1
            comparison_counts[model_b][model_a] += 1

        # Calculate win matrix
        win_matrix = {
            ep_a: {
                ep_b: (
                    win_counts[ep_a][ep_b] / comparison_counts[ep_a][ep_b]
                    if comparison_counts[ep_a][ep_b] > 0
                    else 0.0
                )
                for ep_b in endpoint_names
                if ep_a != ep_b
            }
            for ep_a in endpoint_names
        }

        # Calculate win rates
        win_rates = {
            ep: (
                sum(win_counts[ep].values()) / sum(comparison_counts[ep].values())
                if sum(comparison_counts[ep].values()) > 0
                else 0.0
            )
            for ep in endpoint_names
        }

        rankings = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)

        return EvaluationResult(
            rankings=rankings,
            win_rates=win_rates,
            win_matrix=win_matrix,
            best_pipeline=rankings[0][0] if rankings else "",
            total_queries=len(self._responses),
            total_comparisons=len(grader_results),
        )

    async def evaluate(
        self,
        queries: Optional[List[GeneratedQuery]] = None,
        rubrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Run complete evaluation pipeline with checkpoint support.

        Args:
            queries: Optional pre-generated queries
            rubrics: Optional pre-generated rubrics

        Returns:
            EvaluationResult with rankings
        """
        # Try to resume from checkpoint
        checkpoint = None
        if self._resume:
            checkpoint = self._checkpoint_mgr.load()
        
        # Step 1: Generate or load queries
        if queries:
            self._queries = queries
            logger.info(f"Using {len(queries)} provided queries")
        elif checkpoint and checkpoint.stage.value >= EvaluationStage.QUERIES_GENERATED.value:
            self._queries = self._checkpoint_mgr.load_queries()
            logger.info(f"Resumed {len(self._queries)} queries from checkpoint")
        elif not self._queries:
            await self.generate_queries()
            # Save checkpoint
            self._checkpoint_mgr.save_queries(self._queries)
            self._checkpoint_mgr.update_stage(
                EvaluationStage.QUERIES_GENERATED,
                total_queries=len(self._queries),
                queries_file=str(self._checkpoint_mgr.output_dir / "queries.json"),
            )

        # Step 2: Collect or load responses
        if checkpoint and checkpoint.stage.value >= EvaluationStage.RESPONSES_COLLECTED.value:
            self._responses = self._checkpoint_mgr.load_responses()
            logger.info(f"Resumed {len(self._responses)} responses from checkpoint")
        elif not self._responses:
            await self.collect_responses()
            # Save checkpoint
            self._checkpoint_mgr.save_responses(self._responses)
            self._checkpoint_mgr.update_stage(
                EvaluationStage.RESPONSES_COLLECTED,
                collected_responses=len(self._responses),
                responses_file=str(self._checkpoint_mgr.output_dir / "responses.json"),
            )

        # Step 3: Generate or load rubrics
        if rubrics:
            self._rubrics = rubrics
            logger.info(f"Using {len(rubrics)} provided rubrics")
        elif checkpoint and checkpoint.stage.value >= EvaluationStage.RUBRICS_GENERATED.value:
            self._rubrics = self._checkpoint_mgr.load_rubrics()
            logger.info(f"Resumed {len(self._rubrics)} rubrics from checkpoint")
        elif not self._rubrics:
            await self.generate_rubrics()
            # Save checkpoint
            self._checkpoint_mgr.save_rubrics(self._rubrics)
            self._checkpoint_mgr.update_stage(
                EvaluationStage.RUBRICS_GENERATED,
                rubrics_file=str(self._checkpoint_mgr.output_dir / "rubrics.json"),
            )

        # Step 4: Run pairwise evaluation
        logger.info("Step 4: Running pairwise evaluation...")
        dataset, endpoint_names = self._prepare_pairwise_data(self._responses)

        if not dataset:
            raise ValueError("No valid comparison pairs. Check if responses were collected successfully.")

        grader_results = await self._run_pairwise_evaluation(dataset, self._rubrics)

        # Step 5: Analyze results
        logger.info("Step 5: Analyzing results...")
        result = self._analyze_results(dataset, grader_results, endpoint_names)

        # Mark evaluation complete
        self._checkpoint_mgr.update_stage(
            EvaluationStage.EVALUATION_COMPLETE,
            total_pairs=len(dataset),
            evaluated_pairs=len(grader_results),
        )

        self._display_results(result)
        return result

    def _display_results(self, result: EvaluationResult) -> None:
        """Display evaluation results with formatted output."""
        endpoint_names = list(self.config.target_endpoints.keys())

        # Header
        logger.info("\n" + "=" * 60)
        logger.info("ZERO-SHOT EVALUATION RESULTS")
        logger.info("=" * 60)

        # Summary
        logger.info(f"Task: {self.config.task.description[:50]}...")
        logger.info(f"Queries: {result.total_queries}")
        logger.info(f"Comparisons: {result.total_comparisons}")

        # Rankings
        logger.info("\nRankings:")
        for rank, (name, win_rate) in enumerate(result.rankings, 1):
            bar_len = int(win_rate * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            logger.info(f"  {rank}. {name:<20} [{bar}] {win_rate:.1%}")

        # Win Matrix
        if len(endpoint_names) > 1:
            logger.info("\nWin Matrix (row vs column):")
            # Header row
            max_name_len = max(len(n) for n in endpoint_names)
            header = " " * (max_name_len + 3) + "".join(f"{n[:8]:<10}" for n in endpoint_names)
            logger.info(f"  {header}")

            # Data rows
            for ep_a in endpoint_names:
                row = f"  {ep_a:<{max_name_len}} | "
                for ep_b in endpoint_names:
                    if ep_a == ep_b:
                        row += f"{'--':<10}"
                    else:
                        win_rate = result.win_matrix.get(ep_a, {}).get(ep_b, 0.0)
                        row += f"{win_rate:<10.1%}"
                logger.info(row)

        # Best pipeline
        logger.info(f"\nBest Pipeline: {result.best_pipeline}")
        logger.info("=" * 60)

    def save_results(
        self,
        result: EvaluationResult,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save evaluation results to file.

        Args:
            result: Evaluation result
            output_dir: Output directory (uses config default if not provided)

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir or self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "evaluation_results.json"

        data = {
            "result": result.model_dump(),
            "config": {
                "task": self.config.task.model_dump(),
                "target_endpoints": list(self.config.target_endpoints.keys()),
                "num_queries": self.config.query_generation.num_queries,
            },
            "queries": [q.model_dump() for q in self._queries],
            "rubrics": self._rubrics,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")
        return output_file

