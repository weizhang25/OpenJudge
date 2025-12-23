# Format Graders

Format graders for evaluating structural and formatting aspects of AI responses. These graders validate JSON structures, check length constraints, detect repetition, and verify specific output formats like reasoning tags.

---

## Overview

| Grader | Purpose | Key Use Case |
|--------|---------|--------------|
| `JsonValidatorGrader` | Validates JSON syntax | Structured output validation |
| `JsonMatchGrader` | Deep JSON structure comparison | API response verification |
| `LengthPenaltyGrader` | Penalizes inappropriate length | Content length control |
| `NgramRepetitionPenaltyGrader` | Detects repetitive patterns | Quality assurance |
| `ReasoningFormatGrader` | Validates reasoning tags | Chain-of-thought formatting |
| `ReasoningToolCallFormatGrader` | Validates tool call format | Agent output validation |

---

## JsonValidatorGrader

Validates whether a response is valid JSON. Essential for ensuring structured outputs can be parsed correctly.

**When to use:**
- Structured data generation
- API response validation
- JSON output requirements
- Format compliance checking

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to validate as JSON |

**Scoring:**
- **1.0**: Valid JSON that can be parsed
- **0.0**: Invalid JSON or parse error

**Example:**

```python
import asyncio
from rm_gallery.core.graders.format.json.json_validator import JsonValidatorGrader

async def main():
    grader = JsonValidatorGrader()

    # Valid JSON
    result = await grader.aevaluate(
        response='{"name": "Alice", "age": 30, "skills": ["Python", "AI"]}',
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "Valid JSON"

    # Invalid JSON
    result = await grader.aevaluate(
        response='{"name": "Alice", "age": 30',  # Missing closing brace
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # Error message

asyncio.run(main())
```

---

## JsonMatchGrader

Performs deep structural comparison of JSON objects. Recursively validates that two JSON structures match according to configurable rules.

**When to use:**
- Ground truth comparison for JSON outputs
- API response verification
- Structured data evaluation
- Testing JSON generation accuracy

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reference_response` | str | Yes | Reference JSON string |
| `response` | str | Yes | Generated JSON to compare |
| `strict_order` | bool | No | Whether list order matters (default: True) |
| `ignore_extra_keys` | bool | No | Ignore extra keys in response (default: False) |

**Scoring:**
- **1.0**: JSON structures match completely
- **0.0**: Structures differ or parse error

**Example:**

```python
import asyncio
from rm_gallery.core.graders.format.json.json_match import JsonMatchGrader

async def main():
    # Strict matching
    grader = JsonMatchGrader(strict_order=True)

    result = await grader.aevaluate(
        reference_response='{"name": "Alice", "hobbies": ["reading", "swimming"]}',
        response='{"name": "Alice", "hobbies": ["reading", "swimming"]}',
    )

    print(f"Score: {result.score}")   # 1.0 - exact match

    # Order-independent matching
    grader = JsonMatchGrader(strict_order=False)

    result = await grader.aevaluate(
        reference_response='{"hobbies": ["reading", "swimming"]}',
        response='{"hobbies": ["swimming", "reading"]}',
    )

    print(f"Score: {result.score}")   # 1.0 - matches despite different order

    # Ignore extra keys
    grader = JsonMatchGrader(ignore_extra_keys=True)

    result = await grader.aevaluate(
        reference_response='{"name": "Alice"}',
        response='{"name": "Alice", "age": 30, "city": "NYC"}',
    )

    print(f"Score: {result.score}")   # 1.0 - extra keys ignored

asyncio.run(main())
```

---

## LengthPenaltyGrader

Applies penalties to responses that are too short or too long. Useful for controlling output verbosity.

**When to use:**
- Enforcing response length constraints
- Penalizing overly verbose outputs
- Ensuring minimum content length
- Training models for concise responses

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to evaluate |
| `min_length` | int | No | Minimum acceptable length (default: 10) |
| `max_length` | int | No | Maximum acceptable length (default: 1000) |
| `penalty_rate` | float | No | Penalty per character violation (default: 0.01) |

**Scoring:**
- **0.0**: Length within acceptable range
- **< 0.0**: Negative penalty proportional to length violation

**Penalty calculation:**
- If `length < min_length`: penalty = -(min_length - length) × penalty_rate
- If `length > max_length`: penalty = -(length - max_length) × penalty_rate

**Example:**

```python
import asyncio
from rm_gallery.core.graders.format.length_penalty import LengthPenaltyGrader

async def main():
    grader = LengthPenaltyGrader(
        min_length=50,
        max_length=200,
        penalty_rate=0.1,
    )

    # Acceptable length
    result = await grader.aevaluate(
        response="This response has an acceptable length that falls within the specified range.",
    )

    print(f"Score: {result.score}")   # 0.0 - no penalty
    print(f"Reason: {result.reason}") # "Length acceptable: 50 <= 83 <= 200"

    # Too short
    result = await grader.aevaluate(response="Short")

    print(f"Score: {result.score}")   # -4.5 = -(50-5) * 0.1
    print(f"Reason: {result.reason}") # "Too short: 5 < 50"

    # Too long
    long_text = "A" * 250
    result = await grader.aevaluate(response=long_text)

    print(f"Score: {result.score}")   # -5.0 = -(250-200) * 0.1
    print(f"Reason: {result.reason}") # "Too long: 250 > 200"

asyncio.run(main())
```

---

## NgramRepetitionPenaltyGrader

Detects and penalizes repetitive patterns in text using N-gram analysis. Supports multiple languages and tokenization methods.

**When to use:**
- Detecting repetitive content
- Quality control for generated text
- Training models to avoid repetition
- Evaluating text diversity

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to analyze |
| `n` | int | No | N-gram size (default: 3) |
| `penalty_threshold` | float | No | Threshold for hard penalty (default: 0.3) |
| `penalty_rate` | float | No | Penalty rate per repetition (default: 1.0) |
| `use_soft_penalty` | bool | No | Use soft penalty mode (default: False) |
| `max_penalty` | float | No | Maximum penalty value (default: -1.0) |
| `tokenizer_type` | str | No | Tokenizer type: tiktoken, jieba, simple (default: tiktoken) |
| `analyze_scope` | str | No | Analyze "thought" or "full" text (default: full) |

**Scoring:**
- **0.0**: No significant repetition detected
- **< 0.0**: Negative penalty proportional to repetition rate

**Example:**

```python
import asyncio
from rm_gallery.core.graders.format.ngram_repetition_penalty import NgramRepetitionPenaltyGrader

async def main():
    # Hard threshold penalty
    grader = NgramRepetitionPenaltyGrader(
        n=3,
        penalty_threshold=0.3,
        penalty_rate=1.0,
    )

    # Diverse text
    result = await grader.aevaluate(
        response="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    )

    print(f"Score: {result.score}")   # 0.0 or small penalty
    print(f"Metadata: {result.metadata['repetition_rate']}")

    # Repetitive text
    result = await grader.aevaluate(
        response="This is a test. This is a test. This is a test. This is a test.",
    )

    print(f"Score: {result.score}")   # Large negative penalty
    print(f"Repetition rate: {result.metadata['repetition_rate']:.2f}")

    # Soft penalty mode
    grader = NgramRepetitionPenaltyGrader(
        n=2,
        use_soft_penalty=True,
        max_penalty=-2.0,
        min_scaling=0.2,
    )

    result = await grader.aevaluate(
        response="Different words create different patterns without repetition here.",
    )

    print(f"Score: {result.score}")   # Gradual penalty

asyncio.run(main())
```

---

## ReasoningFormatGrader

Validates that responses follow a specific reasoning format with `<think>` and `<answer>` tags. Essential for chain-of-thought evaluation.

**When to use:**
- Chain-of-thought (CoT) formatting
- Reasoning process validation
- Training models with structured reasoning
- Ensuring proper thought-answer separation

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to validate |
| `think_token` | str | No | Thinking tag name (default: "think") |
| `answer_token` | str | No | Answer tag name (default: "answer") |

**Scoring:**
- **1.0**: Both `<think>` and `<answer>` tags present
- **0.0**: Missing one or both required tags

**Example:**

```python
import asyncio
from rm_gallery.core.graders.format.reasoning_format import ReasoningFormatGrader

async def main():
    grader = ReasoningFormatGrader()

    # Valid format
    result = await grader.aevaluate(
        response="""<think>
First, I need to analyze the problem.
The user is asking about Python benefits.
</think>

<answer>
Python is easy to learn, has extensive libraries, and strong community support.
</answer>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "All format requirements met"

    # Invalid format - missing tags
    result = await grader.aevaluate(
        response="Python is a great programming language for beginners.",
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # "Missing <think></think> tags; Missing <answer></answer> tags"

    # Custom tags
    grader = ReasoningFormatGrader(think_token="reasoning", answer_token="solution")

    result = await grader.aevaluate(
        response="<reasoning>My thought process</reasoning>\n<solution>Final answer</solution>",
    )

    print(f"Score: {result.score}")   # 1.0

asyncio.run(main())
```

---

## ReasoningToolCallFormatGrader

Validates that responses follow proper format for tool-calling agents with reasoning. Checks for `<think>` tags combined with either `<answer>` or `<tool_call>` tags, and validates JSON structure in tool calls.

**When to use:**
- Agent output validation
- Tool-calling format enforcement
- Function calling verification
- Multi-step reasoning with tool use

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The text to validate |

**Valid formats:**
1. `<think>...</think>` + `<answer>...</answer>` - Reasoning with final answer
2. `<think>...</think>` + `<tool_call>...</tool_call>` - Reasoning with tool calls

**Tool call JSON requirements:**
- Must contain `name` field (function name)
- Must contain `arguments` field (function arguments)

**Scoring:**
- **1.0**: Valid format with proper tags and JSON structure
- **0.0**: Invalid format, missing tags, or malformed JSON

**Example:**

```python
import asyncio
from rm_gallery.core.graders.format.reasoning_tool_format import ReasoningToolCallFormatGrader

async def main():
    grader = ReasoningToolCallFormatGrader()

    # Valid reasoning + answer format
    result = await grader.aevaluate(
        response="""<think>
The user wants to know the weather. I should provide the current information.
</think>

<answer>
The current temperature is 72°F with clear skies.
</answer>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "Valid <think></think> + <answer></answer> format"

    # Valid reasoning + tool call format
    result = await grader.aevaluate(
        response="""<think>
I need to search for information about Python.
</think>

<tool_call>
{"name": "search", "arguments": {"query": "Python programming language"}}
</tool_call>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Reason: {result.reason}") # "Valid <think></think> + <tool_call></tool_call> format with valid JSON"

    # Multiple tool calls
    result = await grader.aevaluate(
        response="""<think>
I need to gather data from multiple sources.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "New York"}}
</tool_call>

<tool_call>
{"name": "get_news", "arguments": {"topic": "technology"}}
</tool_call>"""
    )

    print(f"Score: {result.score}")   # 1.0
    print(f"Tool calls: {result.metadata['tool_call_count']}")  # 2

    # Invalid format - missing think tag
    result = await grader.aevaluate(
        response="<answer>Direct answer without thinking</answer>",
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # "Missing <think></think> tags"

    # Invalid format - malformed JSON in tool call
    result = await grader.aevaluate(
        response="""<think>Searching</think>
<tool_call>
{invalid json}
</tool_call>"""
    )

    print(f"Score: {result.score}")   # 0.0
    print(f"Reason: {result.reason}") # "Invalid JSON format in <tool_call> tags"

asyncio.run(main())
```

---

## Combining Format Graders

For comprehensive format evaluation, combine multiple format graders using `GradingRunner`:

```python
import asyncio
from rm_gallery.core.graders.format import (
    JsonValidatorGrader,
    LengthPenaltyGrader,
    NgramRepetitionPenaltyGrader,
)
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    grader_configs = {
        "json_valid": GraderConfig(grader=JsonValidatorGrader()),
        "length": GraderConfig(
            grader=LengthPenaltyGrader(min_length=20, max_length=500)
        ),
        "repetition": GraderConfig(
            grader=NgramRepetitionPenaltyGrader(n=3, penalty_threshold=0.3)
        ),
    }

    runner = GradingRunner(grader_configs=grader_configs)

    dataset = [
        {"response": '{"name": "Alice", "skills": ["Python", "Machine Learning"]}'},
    ]

    results = await runner.arun(dataset)

    print(f"JSON Valid: {results['json_valid'][0].score}")
    print(f"Length Penalty: {results['length'][0].score}")
    print(f"Repetition Penalty: {results['repetition'][0].score}")

asyncio.run(main())
```

---

## Next Steps

- [General Graders](general.md) — Evaluate response quality and relevance
- [Text Graders](text.md) — Evaluate text-specific qualities
- [Build Reward for Training](../get_started/build_reward.md) — Combine graders for RLHF rewards






