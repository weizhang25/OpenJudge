# Train Reward Models

Train custom reward models to learn evaluation criteria from preference data—enabling scalable, consistent graders aligned with your specific quality standards.


## Why Train Reward Models?

While LLM judges and code-based graders handle many evaluation scenarios, trained reward models excel when you need to evaluate at scale (>1M queries/month), maintain consistent criteria learned from data, or reduce per-query costs by 10x versus API judges. Training is ideal when you have labeled preference data (1K-100K examples) and need production-grade evaluation optimized for high-volume deployment.


## Training Approaches

RM-Gallery supports multiple training paradigms through the VERL framework. Choose the approach that matches your data type and evaluation goals:

| Approach | Best For | Training Signal | Complexity |
|----------|----------|-----------------|------------|
| [Bradley-Terry](bradley_terry.md) | Preference pairs | Binary comparisons (chosen/rejected) | Low |
| [SFT](sft.md) | Quick initialization | Multi-turn conversations | Low |
| [Generative (Pointwise)](generative_pointwise.md) | Absolute scoring | Direct score labels (0-4) | Medium |
| [Generative (Pairwise)](generative_pairwise.md) | Comparative ranking | Preference decisions (A/B/tie) | Medium |

!!! tip "How to Choose"
    - **Have preference pairs?** → Use [Bradley-Terry](bradley_terry.md) (simplest, most common)
    - **Starting from scratch?** → Use [SFT](sft.md) first, then fine-tune with another method
    - **Have absolute scores?** → Use [Generative Pointwise](generative_pointwise.md) (e.g., HelpSteer2 ratings)
    - **Have comparison labels?** → Use [Generative Pairwise](generative_pairwise.md) (e.g., "A is better than B")


## Training Architecture

All training methods use the **VERL** (Versatile Efficient Reinforcement Learning) framework with distributed training support:

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
│  ├─ FSDP (Distributed Training)                │
│  ├─ Ray (Resource Management)                  │
│  └─ Multi-GPU/Multi-Node Support               │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Trained Reward Model                           │
│  └─ Integrate as rm_gallery.core grader        │
└─────────────────────────────────────────────────┘
```

> **Note:** Training code (`tutorials/cookbooks/training_reward_model/`) is independent from the core evaluation framework (`rm_gallery.core`). Train models separately, then integrate them as graders.


## Quick Start

Get started with Bradley-Terry training, the most common approach for learning from preference pairs:

```bash
# 1. Install training dependencies
pip install verl torch transformers datasets ray

# 2. Prepare your data (Parquet format)
# Each row: {"chosen": "better response", "rejected": "worse response"}

# 3. Run training
cd tutorials/cookbooks/training_reward_model/bradley_terry
bash run_bt.sh
```

After training completes, use your model as a grader:

```python
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common import RelevanceGrader

# Load trained model
model = OpenAIChatModel(model="./checkpoints/your-model", is_local=True)
grader = RelevanceGrader(model=model)

# Evaluate responses
result = await grader.aevaluate(query="...", response="...")
```

**For detailed setup, data formats, and training configurations, see individual training method pages.**


## Data Requirements

All training approaches require data in **Parquet format**. Export your evaluation data using RM-Gallery utilities:

```python
from rm_gallery.core.generator import export_data

export_data(
    eval_cases=your_data,
    output_dir="./training_data",
    formats=["parquet"],
    split_ratio={"train": 0.8, "test": 0.2}
)
```

**Required columns vary by training method:**

- **Bradley-Terry**: `chosen`, `rejected` (text strings)
- **SFT**: `messages` (conversation format)
- **Generative Pointwise**: `input`, `output`, `label` (structured JSON)
- **Generative Pairwise**: `input`, `output`, `metadata` (structured JSON)

See individual training pages for complete data format specifications and examples.


## Next Steps

**Choose Your Training Method:**

- [Bradley-Terry Training](bradley_terry.md) — Train with binary preference pairs (most common)
- [SFT for Reward Models](sft.md) — Initialize with supervised fine-tuning
- [Generative Pointwise](generative_pointwise.md) — Train with absolute scores (HelpSteer2-style)
- [Generative Pairwise](generative_pairwise.md) — Train with comparative preferences

**Alternative Approaches:**

- [Create Custom Graders](../create_custom_graders.md) — Build LLM/code-based graders without training
- [Generate from Data](../generate_graders_from_data.md) — Auto-generate rubrics from examples






