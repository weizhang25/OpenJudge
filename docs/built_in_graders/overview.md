# Built-in Graders Overview

RM-Gallery provides **50+ pre-built graders** for evaluating AI responses across quality dimensions, agent behaviors, formats, and modalities. All graders are **rigorously evaluated** on benchmark datasets to ensure reliability and accuracy.



## Quick Start

Get started with RM-Gallery graders in three simple steps: initialize a model, create a grader, and evaluate responses. All graders follow a consistent API design for easy integration.

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common import RelevanceGrader

async def main():
    # Initialize model and grader
    model = OpenAIChatModel(model="qwen3-32b")
    grader = RelevanceGrader(model=model)
    
    # Evaluate a response
    result = await grader.aevaluate(
        query="What is machine learning?",
        response="Machine learning is a subset of AI that enables systems to learn from data."
    )
    
    print(f"Score: {result.score}")  # e.g., 4.5
    print(f"Reason: {result.reason}")  # Detailed explanation

asyncio.run(main())
```

!!! tip "Understanding Results"
    All graders return a result object with `score`, `reason`, and `metadata` fields. Score ranges vary by grader category (see table below).

For detailed installation and environment setup, see the [Quick Start Guide](../get_started/quickstart.md).


## Available Graders

Choose the right grader for your evaluation needs. RM-Gallery organizes graders by evaluation focus, making it easy to find graders for specific tasks. Each category includes graders optimized for different aspects of AI evaluation:

**Selection Guide:**

- Use **General** graders for basic quality checks (relevance, safety, correctness)
- Use **Agent** graders when evaluating autonomous agents and tool-calling behaviors
- Use **Text** graders for fast, zero-cost similarity and matching tasks
- Use **Code & Math** graders for programming and mathematical problem evaluation
- Use **Format** graders to validate structured outputs (JSON, length, patterns)
- Use **Multimodal** graders for vision-language tasks and image evaluation

| Category | Implementation | Score Range | Count | Key Graders |
|----------|---------------|-------------|-------|-------------|
| **[General](general.md)** | LLM-Based | 1-5 | 5 | Relevance, Hallucination, Harmfulness, Instruction Following, Correctness |
| **[Agent](agent_graders.md)** | LLM-Based | 0-1 | 15+ | Action Alignment, Tool Selection, Memory Accuracy, Plan Feasibility, Reflection |
| **[Text](text.md)** | Code-Based | 0-1 | 3 | Similarity (15+ algorithms), String Match, Number Accuracy |
| **[Code & Math](code_math.md)** | Code-Based | 0-1 | 5 | Code Execution, Syntax Check, Code Style, Patch Similarity, Math Verify |
| **[Format](format.md)** | Code-Based | 0-1 | 6 | JSON Validator, JSON Match, Length Penalty, N-gram Repetition, Reasoning Format |
| **[Multimodal](multimodal.md)** | LLM-Based | 0-1 | 3 | Image Coherence, Image Helpfulness, Text-to-Image |

!!! tip "Implementation Types"
    - **LLM-Based** graders (General, Agent, Multimodal): Nuanced quality assessment using LLM judges
    - **Code-Based** graders (Text, Code, Format): Fast, deterministic, zero-cost evaluation using algorithms


## Next Steps

**Explore Graders by Category:**

- [General Graders](general.md) — Quality assessment (Relevance, Hallucination, Harmfulness, Instruction Following, Correctness)
- [Agent Graders](agent_graders.md) — Agent evaluation (Action, Tool, Memory, Plan, Reflection, Trajectory)
- [Text Graders](text.md) — Text similarity and matching (15+ algorithms)
- [Code & Math Graders](code_math.md) — Code execution and math verification
- [Format Graders](format.md) — Structure validation (JSON, Length, Repetition, Reasoning Format)
- [Multimodal Graders](multimodal.md) — Vision and image tasks

**Advanced Usage:**

- [Run Evaluation Tasks](../running_graders/run_tasks.md) — Batch processing and reporting
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build domain-specific evaluators
