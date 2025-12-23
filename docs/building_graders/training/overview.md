# Training Graders Overview

Train custom reward models and graders to better align with your specific evaluation needs. This guide covers different training approaches for building effective reward models.

---

## Why Train Custom Graders?

Pre-built graders work well for general tasks, but custom training enables:

!!! tip "Benefits of Custom Training"
    - **Domain Adaptation**: Align evaluations with your specific use case
    - **Preference Learning**: Capture nuanced human preferences from your data
    - **Cost Reduction**: Replace expensive API-based judges with self-hosted models
    - **Consistency**: Maintain stable evaluation criteria across your application

---

## Training Approaches

RM-Gallery supports multiple training paradigms through the VERL framework. Choose the approach that matches your data type and evaluation goals:

| Approach | Best For | Training Signal | Complexity |
|----------|----------|-----------------|------------|
| [Bradley-Terry](bradley_terry.md) | Preference pairs | Binary comparisons (chosen/rejected) | Low |
| [Generative (Pointwise)](generative_pointwise.md) | Absolute scoring | Direct score labels (0-4) | Medium |
| [Generative (Pairwise)](generative_pairwise.md) | Comparative ranking | Preference decisions (A/B/tie) | Medium |
| [SFT](sft.md) | Quick initialization | Multi-turn conversations | Low |

!!! tip "How to Choose"
    - **Have preference pairs?** → Use [Bradley-Terry](bradley_terry.md) (simplest, most common)
    - **Have absolute scores?** → Use [Generative Pointwise](generative_pointwise.md) (e.g., HelpSteer2 ratings)
    - **Have comparison labels?** → Use [Generative Pairwise](generative_pairwise.md) (e.g., "A is better than B")
    - **Starting from scratch?** → Use [SFT](sft.md) first, then fine-tune with another method

---

## Training Architecture

All training methods use the **VERL** (Versatile Efficient Reinforcement Learning) framework with:

- **FSDP** (Fully Sharded Data Parallel) for distributed training
- **Ray** for resource management and job scheduling
- **Multi-GPU/Multi-Node** support for scalability
- **Mixed Precision** (BF16) for efficiency

**Framework Integration:**

```
┌─────────────────────────────────────────────────┐
│  Training Data (Parquet)                        │
│  ├─ Preference pairs (Bradley-Terry)            │
│  ├─ Scored responses (Pointwise)                │
│  └─ Comparison labels (Pairwise)                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  VERL Framework                                 │
│  ├─ Dataset Loader (BaseTrainDataset)          │
│  ├─ Distributed Training (FSDP/Ray)            │
│  └─ Reward Function (Custom scoring)           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Trained Model                                  │
│  └─ Use in rm_gallery.core evaluation          │
└─────────────────────────────────────────────────┘
```

> **Note:** Training code (`tutorials/cookbooks/training_reward_model/`) is independent from the core evaluation framework (`rm_gallery.core`). Train models separately, then integrate them as graders.

---

## Data Requirements & Preparation

All training approaches use **Parquet files** with specific column structures. Use RM-Gallery's data export utilities to convert your evaluation data to the required format.

### Data Format by Training Method

=== "Bradley-Terry"

    ```python
    {
        "chosen": "High-quality response text",
        "rejected": "Lower-quality response text"
    }
    ```

=== "Pointwise"

    ```python
    {
        "input": [{"role": "user", "content": "query"}],
        "output": {"content": "response", "label": {"helpfulness": 3}},
        "data_source": "helpsteer2"
    }
    ```

=== "Pairwise"

    ```python
    {
        "input": [{"role": "user", "content": "query"}],
        "output": [
            {"content": "response_a"},
            {"content": "response_b"}
        ],
        "metadata": {"preferred": "A", "preference_strength": 2}
    }
    ```

### Export Your Data

Use RM-Gallery's data export utilities:

```python
from rm_gallery.core.generator import export_data

# Export from evaluation cases
export_data(
    eval_cases=your_evaluation_data,
    output_dir="./training_data",
    formats=["parquet"],
    split_ratio={"train": 0.8, "test": 0.2}
)
```

---

## Quick Start

### 1. Prerequisites

Install training dependencies:

```bash
# Install VERL framework
pip install verl

# Install training requirements
pip install torch transformers datasets ray
```

### 2. Convert Data to Parquet

Use the command-line tool to convert your data:

```bash
# Example: Export HelpSteer2 dataset
python -m rm_gallery.core.generator.export \
    --dataset helpsteer2 \
    --output-dir ./data \
    --format parquet
```

### 3. Choose Training Method

Select based on your data and requirements:

=== "Bradley-Terry"

    ```bash
    # Simplest, binary preferences
    cd tutorials/cookbooks/training_reward_model/bradley_terry
    bash run_bt.sh
    ```

=== "Generative Pointwise"

    ```bash
    # Absolute scores
    cd tutorials/cookbooks/training_reward_model/generative/pointwise
    bash run_pointwise.sh
    ```

=== "Generative Pairwise"

    ```bash
    # Comparison preferences
    cd tutorials/cookbooks/training_reward_model/generative/pairwise
    bash run_pairwise.sh
    ```

=== "SFT"

    ```bash
    # Supervised fine-tuning first
    cd tutorials/cookbooks/training_reward_model/sft
    bash sft_rm.sh
    ```

### 4. Build Grader from Trained Model

Use your trained model to construct a grader:

```python
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common import RelevanceGrader

# Load your trained model
model = OpenAIChatModel(
    model="./checkpoints/your-trained-model",
    is_local=True
)

# Use as a grader
grader = RelevanceGrader(model=model)
result = await grader.aevaluate(
    query="Your query",
    response="Model response"
)
```

---

## Next Steps

- [Bradley-Terry Training](bradley_terry.md) — Train with binary preference pairs
- [Generative Pointwise](generative_pointwise.md) — Train with absolute scores
- [Generative Pairwise](generative_pairwise.md) — Train with comparative preferences
- [SFT for Reward Models](sft.md) — Initialize with supervised fine-tuning






