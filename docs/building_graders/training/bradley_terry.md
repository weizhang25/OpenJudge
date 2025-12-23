# Bradley-Terry Training

Train reward models using Bradley-Terry loss on preference pairs. This approach learns to rank responses by modeling the probability that one response is preferred over another.

---

## Overview

Bradley-Terry training is the **simplest and most widely used** method for reward model training. It works with binary preference data (chosen vs. rejected) and optimizes the model to predict which response humans prefer.

!!! tip "When to use Bradley-Terry"
    - You have preference pairs (e.g., from RLHF annotation)
    - Binary comparison data (response A better than response B)
    - Want a simple, stable training process
    - Need a model that outputs scalar reward scores

**Training objective:**

The model learns to maximize:

\[
\mathcal{L} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})
\]

Where \( r \) is the reward score and \( \sigma \) is the sigmoid function.

---

## Data Format

### Required Structure

Bradley-Terry training expects Parquet files with two columns:

| Column | Type | Description |
|--------|------|-------------|
| `chosen` | str | Text of the preferred response |
| `rejected` | str | Text of the lower-quality response |

**Example:**

```python
{
    "chosen": "Regular exercise improves cardiovascular health, boosts mood, and increases energy levels. It also helps maintain healthy weight and reduces disease risk.",
    "rejected": "Exercise is good for you."
}
```

### Data Preparation

=== "Option 1: Use existing datasets"

    ```bash
    # HuggingFace datasets with preference pairs
    # Example: hendrydong/preference_700K
    TRAIN_FILE=hendrydong/preference_700K/train.parquet
    VAL_FILE=hendrydong/preference_700K/test.parquet
    ```

=== "Option 2: Convert from RM-Gallery evaluation data"

    ```python
    import pandas as pd
    from rm_gallery.core.generator import export_data

    # Create preference pairs from scored data
    def create_preference_pairs(eval_cases):
        pairs = []
        for case in eval_cases:
            # Assuming cases have response_a, response_b with scores
            if case['score_a'] > case['score_b']:
                pairs.append({
                    'chosen': case['response_a'],
                    'rejected': case['response_b']
                })
            elif case['score_b'] > case['score_a']:
                pairs.append({
                    'chosen': case['response_b'],
                    'rejected': case['response_a']
                })
        return pairs

    # Export to parquet
    df_train = pd.DataFrame(create_preference_pairs(train_data))
    df_train.to_parquet('./data/bt_train.parquet')

    df_val = pd.DataFrame(create_preference_pairs(val_data))
    df_val.to_parquet('./data/bt_val.parquet')
    ```

---

## Training Setup

### 1. Configuration

The Bradley-Terry trainer uses Hydra configuration from `trainer.yaml`:

```yaml
# trainer.yaml
model:
  partial_pretrain: Qwen/Qwen2.5-7B-Instruct
  trust_remote_code: true
  enable_gradient_checkpointing: true
  strategy: fsdp  # or fsdp2 for PyTorch >= 2.4
  fsdp_config:
    cpu_offload: false
    model_dtype: bf16

data:
  train_files: ./data/train.parquet
  val_files: ./data/test.parquet
  max_length: 4096
  micro_batch_size_per_gpu: 1
  train_batch_size: 256
  filter_overlong_prompts: true

optim:
  lr: 5e-7
  betas: [0.9, 0.95]
  weight_decay: 0.1
  clip_grad: 2.0
  lr_scheduler: cosine
  warmup_steps_ratio: 0.05

trainer:
  total_epochs: 3
  save_freq: 500
  test_freq: 500
  logger: ['console', 'swanlab']
  project_name: rm-gallery-bt
  experiment_name: qwen2.5-7b-bt
  default_local_dir: ./checkpoints/bt
```

### 2. Launch Training Script

=== "Single-node, multi-GPU"

    ```bash
    cd tutorials/cookbooks/training_reward_model/bradley_terry

    # Edit run_bt.sh with your configuration
    export N_GPUS_PER_NODE=8

    bash run_bt.sh
    ```

=== "Multi-node setup"

    ```bash
    # On master node (NODE_RANK=0)
    export MASTER_ADDR=192.168.1.10
    export MASTER_PORT=29500
    export NNODES=4
    export NODE_RANK=0
    export N_GPUS_PER_NODE=8

    bash run_bt.sh

    # On worker nodes (NODE_RANK=1,2,3...)
    export MASTER_ADDR=192.168.1.10
    export MASTER_PORT=29500
    export NNODES=4
    export NODE_RANK=1  # Change for each node
    export N_GPUS_PER_NODE=8

    bash run_bt.sh
    ```

---

## Script Configuration

### Training Script (run_bt.sh)

```bash
#!/bin/bash
set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# Distributed training configuration
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

# Model and data configuration
SAVE_PATH=./checkpoints/bt
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
TRAIN_FILE=hendrydong/preference_700K/train.parquet
VAL_FILE=hendrydong/preference_700K/test.parquet

PROJECT_NAME=rm-gallery-bt
EXPERIMENT_NAME=qwen2.5-7b-bt-${TIMESTAMP}

# Launch training with torchrun
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./trainer.py \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    model.partial_pretrain=$MODEL_PATH \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=256 \
    optim.lr=5e-7 \
    optim.clip_grad=2 \
    trainer.total_epochs=3 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.logger=['console','swanlab'] \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$SAVE_PATH/$EXPERIMENT_NAME
```

### Key Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `MODEL_PATH` | Base model for initialization | `Qwen/Qwen2.5-7B-Instruct` |
| `TRAIN_FILE` | Training data path | Parquet file or HF dataset |
| `VAL_FILE` | Validation data path | Parquet file or HF dataset |
| `data.train_batch_size` | Global batch size | 256 (adjust for GPU memory) |
| `data.micro_batch_size_per_gpu` | Per-GPU micro batch | 1-2 (increase if memory allows) |
| `optim.lr` | Learning rate | 5e-7 to 1e-6 |
| `trainer.total_epochs` | Training epochs | 2-3 (more may overfit) |

---

## Custom Dataset

### Creating a Custom Dataset Class

If your data has a different structure, create a custom dataset:

```python
# custom_bt_dataset.py
from typing import Any, Dict, List
from tutorials.cookbooks.training_reward_model.base import BaseTrainDataset

class CustomBTDataset(BaseTrainDataset):
    """Custom Bradley-Terry dataset with specialized preprocessing"""

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build chat messages from your data format"""
        # Customize based on your data structure
        query = example.get('query', '')
        return [
            {"role": "user", "content": query}
        ]

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply tokenizer chat template"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_ground_truth(self, row_dict: Dict[str, Any]) -> str:
        """Extract ground truth label"""
        return row_dict.get('label', '')

    def _get_data_source(self, row_dict: Dict[str, Any]) -> str:
        """Get data source identifier"""
        return row_dict.get('source', 'custom')
```

### Using Custom Dataset

Update `trainer.yaml` to use your custom class:

```yaml
data:
  custom_cls:
    path: ./custom_bt_dataset.py
    name: CustomBTDataset
```

---

## Monitoring Training

### Metrics

Bradley-Terry training tracks:

- **train/loss**: Bradley-Terry loss on training set
- **train/accuracy**: % of correct preference predictions
- **train/lr**: Current learning rate
- **val/loss**: Validation loss
- **val/accuracy**: Validation preference accuracy

### Expected Training Curve

!!! example "Typical Training Progress"
    ```
    Epoch 1: loss ~0.6, accuracy ~65%
    Epoch 2: loss ~0.4, accuracy ~75%
    Epoch 3: loss ~0.3, accuracy ~80%
    ```
    
    Higher accuracy indicates the model correctly predicts human preferences.

### Logging to WandB/SwanLab

```bash
# Enable experiment tracking
trainer.logger=['console','wandb']
trainer.project_name=my-reward-model
trainer.experiment_name=qwen-bt-run1
```

---

## Using Trained Models

### 1. Load as OpenAI-Compatible Model

```python
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(
    model="./checkpoints/bt/qwen2.5-7b-bt/global_step_3000",
    is_local=True
)
```

### 2. Use in Graders

```python
from rm_gallery.core.graders.common import RelevanceGrader

grader = RelevanceGrader(model=model)

result = await grader.aevaluate(
    query="What are the benefits of exercise?",
    response="Regular exercise improves health and mood."
)

print(f"Score: {result.score}")
```

### 3. Batch Evaluation with GradingRunner

```python
from rm_gallery.core.runner import GradingRunner, GraderConfig

runner = GradingRunner(
    grader_configs={
        "custom_reward": GraderConfig(grader=RelevanceGrader(model=model))
    }
)

results = await runner.arun(your_dataset)
```

---

## Advanced Configuration

=== "FSDP2 (PyTorch >= 2.4)"

    For better performance with newer PyTorch:

    ```yaml
    model:
      strategy: fsdp2
      fsdp_config:
        cpu_offload: false
        param_offload: false
        optimizer_offload: false
    ```

=== "CPU Offloading (Memory-Constrained)"

    ```yaml
    model:
      fsdp_config:
        cpu_offload: true
        offload_params: true
    ```

=== "Gradient Checkpointing (Reduce Memory)"

    ```yaml
    model:
      enable_gradient_checkpointing: true
    ```

=== "Learning Rate Schedulers"

    ```yaml
    optim:
      lr_scheduler: cosine  # or 'wsd' (warmup-stable-decay)
      warmup_steps_ratio: 0.05
    ```

---


## Next Steps

- [Generative Pointwise](generative_pointwise.md) — Train with absolute score labels
- [Generative Pairwise](generative_pairwise.md) — Train with comparative evaluations
- [SFT for Reward Models](sft.md) — Pre-train with supervised fine-tuning
- [Training Overview](overview.md) — Compare all training methods






