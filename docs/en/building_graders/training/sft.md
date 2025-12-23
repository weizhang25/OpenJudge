# Supervised Fine-Tuning for Reward Models

Initialize reward models through supervised fine-tuning (SFT) on multi-turn conversation data. This provides a strong foundation before applying preference learning methods.

---

## Overview

SFT training teaches models to generate fluent, coherent responses through supervised learning on conversation data. While not a reward modeling technique itself, SFT serves as **essential initialization** for more advanced training methods.

**When to use:**
- **Before** Bradley-Terry or generative training
- Building domain-specific conversation models
- Adapting base models to evaluation formats
- Pre-training on instruction-following data

**Benefits:**
- Improves output quality and format consistency
- Reduces training time for downstream tasks
- Enables better instruction following
- Establishes baseline capabilities

**Training objective:**

Standard next-token prediction with masking on user inputs:

\[
\mathcal{L} = -\sum_{t} \log P(y_t | y_{<t}, x)
\]

Where \( y \) are assistant responses and \( x \) is the conversation context.

---

## Data Format

### Required Structure

SFT training expects Parquet files with conversation messages:

| Column | Type | Description |
|--------|------|-------------|
| `messages` | List[Dict] | Multi-turn conversation |

**Example:**

```python
{
    "messages": [
        {
            "role": "user",
            "content": "What are the health benefits of exercise?"
        },
        {
            "role": "assistant",
            "content": "Regular exercise provides numerous health benefits:\n\n1. **Cardiovascular Health**: Strengthens heart and improves circulation\n2. **Mental Well-being**: Releases endorphins, reducing stress and anxiety\n3. **Weight Management**: Burns calories and builds muscle\n4. **Disease Prevention**: Reduces risk of diabetes, heart disease, and certain cancers\n5. **Energy Levels**: Increases stamina and reduces fatigue\n\nAim for at least 150 minutes of moderate exercise per week."
        },
        {
            "role": "user",
            "content": "How do I start if I'm a beginner?"
        },
        {
            "role": "assistant",
            "content": "Here's a beginner-friendly approach:\n\n**Week 1-2**: Start with 15-20 minute walks, 3-4 times per week\n**Week 3-4**: Increase to 30 minutes, add light stretching\n**Week 5+**: Incorporate variety - cycling, swimming, or bodyweight exercises\n\nKey tips:\n- Listen to your body\n- Stay hydrated\n- Warm up and cool down\n- Set realistic goals\n\nConsider consulting a fitness professional for personalized guidance."
        }
    ]
}
```

### Message Roles

Standard conversation roles:

- **user**: User inputs/queries
- **assistant**: Model responses (training targets)
- **system**: Optional system instructions

---

## Data Preparation

### Option 1: From Existing Datasets

Use instruction-following or conversation datasets:

```python
from datasets import load_dataset
import pandas as pd

# Load ShareGPT-style dataset
dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")

# Convert to messages format
def convert_to_messages(example):
    messages = []
    for msg in example["conversations"]:
        role = "assistant" if msg["from"] == "gpt" else "user"
        messages.append({
            "role": role,
            "content": msg["value"]
        })
    return {"messages": messages}

train_data = dataset["train"].map(convert_to_messages)

# Save as parquet
df = pd.DataFrame(train_data)
df[['messages']].to_parquet("./data/sft_train.parquet")
```

### Option 2: Create Evaluation-Style Conversations

Build conversations focused on evaluation tasks:

```python
import pandas as pd

# Evaluation-focused SFT data
conversations = [
    {
        "messages": [
            {
                "role": "user",
                "content": "Evaluate this response for helpfulness:\n\nQuery: What is Python?\nResponse: Python is a programming language.\n\nProvide a score from 0-4."
            },
            {
                "role": "assistant",
                "content": "**Helpfulness Score: 2**\n\nThis response is technically correct but lacks detail. A more helpful answer would explain Python's key features (high-level, interpreted, versatile) and common use cases (web development, data science, automation)."
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Compare these two responses:\n\nQuery: Benefits of exercise?\n\nResponse A: Improves health, mood, energy.\nResponse B: Regular exercise strengthens your cardiovascular system, boosts mental health through endorphin release, increases energy levels, and helps maintain healthy weight.\n\nWhich is better?"
            },
            {
                "role": "assistant",
                "content": "**Preference: B**\n\nResponse B is significantly better because:\n1. More comprehensive coverage\n2. Specific mechanisms (endorphins, cardiovascular)\n3. Better structure and detail\n4. More actionable information\n\nResponse A is too brief and generic."
            }
        ]
    }
]

df = pd.DataFrame(conversations)
df.to_parquet("./data/eval_sft_train.parquet")
```

### Option 3: Domain-Specific Conversations

```python
# Medical evaluation examples
medical_sft = [
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a medical response evaluator. Assess accuracy, clarity, and safety."
            },
            {
                "role": "user",
                "content": "Evaluate this medical advice: 'For headaches, take aspirin.'"
            },
            {
                "role": "assistant",
                "content": "**Safety Score: 3/5**\n\nConcerns:\n- No dosage information\n- Missing contraindications (bleeding disorders, children)\n- Lacks differential diagnosis\n\nBetter response would include: proper dosage, when to seek medical help, and relevant warnings."
            }
        ]
    }
]
```

---

## Training Setup

### Configuration (trainer.yaml)

SFT uses standard supervised training configuration:

```yaml
# Model configuration
model:
  partial_pretrain: Qwen/Qwen2.5-7B-Instruct
  trust_remote_code: true
  enable_gradient_checkpointing: true
  strategy: fsdp  # or fsdp2
  use_liger: false  # Liger kernel optimization
  fsdp_config:
    cpu_offload: false
    model_dtype: bf16
    param_offload: false
    optimizer_offload: false

# Data configuration
data:
  train_files: ./data/train.parquet
  val_files: ./data/test.parquet
  train_batch_size: 96
  micro_batch_size: 12
  max_length: 8192
  truncation: right
  multiturn:
    enable: true
    messages_key: messages
    tools_key: null

# Optimizer configuration
optim:
  lr: 2e-5
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps_ratio: 0.03

# Training configuration
trainer:
  total_epochs: 1
  save_freq: 500
  test_freq: 500
  logger: ['console', 'swanlab']
  project_name: rm-gallery-sft
  experiment_name: qwen-sft
  default_local_dir: ./checkpoints/sft

# Advanced features
ulysses_sequence_parallel_size: 8  # Sequence parallelism
use_remove_padding: true  # Efficient attention
```

---

## Launch Training

### Training Script (sft_rm.sh)

```bash
#!/bin/bash
set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# Distributed training configuration
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-2}
NODE_RANK=${RANK}

echo "=== SFT Training Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "N_NODES: $NNODES"
echo "N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
echo "==================================="

# Model and data paths
SAVE_PATH=./checkpoints/sft
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
TRAIN_FILE=./data/train.parquet
VAL_FILE=./data/test.parquet

PROJECT_NAME=rm-gallery-sft
EXPERIMENT_NAME=sft-${TIMESTAMP}

# Launch with torchrun
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=96 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key="null" \
    data.micro_batch_size=12 \
    data.truncation=right \
    model.enable_gradient_checkpointing=true \
    model.partial_pretrain=$MODEL_PATH \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.model_dtype="bf16" \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$PROJECT_NAME \
    data.max_length=8192 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true
```

### Run Training

```bash
# Single-node training
cd tutorials/cookbooks/training_reward_model/sft
export N_GPUS_PER_NODE=8
bash sft_rm.sh

# Multi-node training
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export NNODES=4
export NODE_RANK=0  # Change per node
bash sft_rm.sh
```

---

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MODEL_PATH` | Base model | `Qwen/Qwen2.5-7B-Instruct` |
| `data.train_batch_size` | Global batch size | 96 |
| `data.micro_batch_size` | Per-GPU micro batch | 12 |
| `data.max_length` | Max sequence length | 8192 |
| `optim.lr` | Learning rate | 2e-5 |
| `trainer.total_epochs` | Training epochs | 1-2 |
| `ulysses_sequence_parallel_size` | Sequence parallelism | 8 |

---

## Advanced Features

### Sequence Parallelism (Ulysses)

For long sequences, use Ulysses sequence parallelism:

```bash
ulysses_sequence_parallel_size=8  # Split sequences across 8 GPUs
data.max_length=16384             # Support longer contexts
```

This enables training on very long conversations efficiently.

### Remove Padding

Optimize attention computation by removing padding:

```bash
use_remove_padding=true  # More efficient attention
```

### Gradient Checkpointing

Reduce memory usage:

```bash
model.enable_gradient_checkpointing=true
```

### Liger Kernel

Optimized kernel for faster training:

```bash
model.use_liger=true
```

---

## Monitoring

### Metrics

Track during SFT:

- **train/loss**: Training loss (should decrease)
- **val/loss**: Validation loss
- **Learning Rate**: Current LR from scheduler
- **Tokens/Second**: Training throughput

### Expected Training Curve

```
Epoch 0.1: loss ~2.5
Epoch 0.3: loss ~1.8
Epoch 0.7: loss ~1.2
Epoch 1.0: loss ~0.9
```

Lower loss indicates better next-token prediction.

### Logging

```bash
# Console logging
trainer.logger=['console']

# Add WandB
trainer.logger=['console','wandb']

# Add SwanLab
trainer.logger=['console','swanlab']
```

---

## Using SFT Models

### 1. Direct Use

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(
    model="./checkpoints/sft/qwen-sft-final",
    is_local=True
)

# Test conversation
messages = [
    {"role": "user", "content": "Evaluate this response..."}
]

result = await model.acomplete(messages)
print(result)
```

### 2. As Initialization for Further Training

**Recommended workflow:**

```bash
# Step 1: SFT training
cd sft
bash sft_rm.sh

# Step 2: Use SFT checkpoint for Bradley-Terry
cd ../bradley_terry
MODEL_PATH=../sft/checkpoints/qwen-sft-final
bash run_bt.sh

# Or use for generative training
cd ../generative/pointwise
MODEL_PATH=../../sft/checkpoints/qwen-sft-final
bash run_pointwise.sh
```

### 3. Evaluation

```python
# Test on held-out conversations
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./checkpoints/sft-final")
model = AutoModelForCausalLM.from_pretrained("./checkpoints/sft-final")

test_messages = [
    {"role": "user", "content": "What is machine learning?"}
]

inputs = tokenizer.apply_chat_template(
    test_messages,
    return_tensors="pt",
    add_generation_prompt=True
)

outputs = model.generate(inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0])
print(response)
```

---

## Best Practices

### 1. Data Quality

**High-quality SFT data is crucial:**

- Use diverse conversation sources
- Include multi-turn dialogues
- Filter low-quality or toxic content
- Balance different conversation types

### 2. Epochs

**Don't overtrain:**

```bash
trainer.total_epochs=1  # Usually sufficient
# More epochs risk overfitting
```

### 3. Sequence Length

**Match your downstream use case:**

```bash
# Evaluation tasks: 4K-8K
data.max_length=4096

# Long document analysis: 16K+
data.max_length=16384
ulysses_sequence_parallel_size=8
```

### 4. Validation

**Monitor validation loss:**

```bash
trainer.test_freq=500  # Validate every 500 steps
```

Stop training if validation loss plateaus or increases.

---

## SFT + Reward Model Training Pipeline

### Complete Workflow

```bash
# 1. Prepare SFT data (multi-turn conversations)
python prepare_sft_data.py \
    --source sharegpt \
    --output ./sft_data

# 2. SFT training
cd tutorials/cookbooks/training_reward_model/sft
bash sft_rm.sh

# 3. Prepare reward data (preferences or scores)
python prepare_reward_data.py \
    --source helpsteer2 \
    --output ./reward_data

# 4. Bradley-Terry training (using SFT checkpoint)
cd ../bradley_terry
MODEL_PATH=../sft/checkpoints/sft-final
bash run_bt.sh

# 5. Final evaluation
python evaluate_pipeline.py \
    --sft-model ../sft/checkpoints/sft-final \
    --reward-model ./checkpoints/bt-final
```

### Why This Works

1. **SFT**: Teaches basic conversation and instruction following
2. **Reward Training**: Specializes model for evaluation/ranking
3. **Result**: High-quality evaluator with strong baseline capabilities

---

## Comparison with Other Methods

| Aspect | SFT | Bradley-Terry | Generative |
|--------|-----|---------------|------------|
| **Data** | Conversations | Preferences | Labeled scores |
| **Objective** | Next-token | Ranking | Structured output |
| **Training Speed** | Fast | Medium | Slower (RL) |
| **Use Case** | Initialization | Ranking | Scoring |
| **Explainability** | Low | Low | High |

**Recommendation:** Use SFT as initialization, then apply specialized reward training.

---

## Next Steps

- [Bradley-Terry Training](bradley_terry.md) — Train with preference pairs after SFT
- [Generative Pointwise](generative_pointwise.md) — Score-based training after SFT
- [Generative Pairwise](generative_pairwise.md) — Comparison training after SFT
- [Training Overview](overview.md) — Complete training strategy guide

