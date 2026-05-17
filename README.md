# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

This project implements the Transformer architecture from "Attention Is All You Need" in PyTorch for German-to-English machine translation on the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Transformer architecture, masking, and configurable attention scaling
├── lr_scheduler.py    # Noam scheduler
├── dataset.py         # Multi30k dataset loading and spaCy tokenization
├── train.py           # Training loop, W&B logging, gradient-norm logging, and checkpointing
```

## Google Colab Setup from GitHub

Replace `YOUR_GITHUB_REPO_URL` with your repository URL.

```python
!git clone YOUR_GITHUB_REPO_URL
%cd da6401_assignment_3_abdallah
```

Install Python dependencies and the spaCy tokenizers used by `dataset.py`:

```python
!pip install -r requirements.txt
!python -m spacy download de_core_news_sm
!python -m spacy download en_core_web_sm
```

Log in to Weights & Biases:

```python
import wandb
wandb.login()
```

## Task 2.1: Noam Scheduler vs Fixed LR

Run the Noam scheduler condition. Because this implementation multiplies the Noam schedule by the optimizer base LR, use `--lr 1.0` for the Noam run.

```python
!python train.py \
  --project da6401-a3 \
  --task task2.1 \
  --run_name task2_1_noam \
  --scheduler noam \
  --attention_scaling scaled \
  --epochs 20 \
  --batch_size 64 \
  --d_model 256 \
  --layers 4 \
  --heads 8 \
  --d_ff 1024 \
  --dropout 0.1 \
  --warmup_steps 4000 \
  --lr 1.0 \
  --seed 42
```

Run the fixed learning-rate condition:

```python
!python train.py \
  --project da6401-a3 \
  --task task2.1 \
  --run_name task2_1_fixed_lr_1e-4 \
  --scheduler fixed \
  --attention_scaling scaled \
  --epochs 20 \
  --batch_size 64 \
  --d_model 256 \
  --layers 4 \
  --heads 8 \
  --d_ff 1024 \
  --dropout 0.1 \
  --warmup_steps 4000 \
  --lr 1e-4 \
  --seed 42
```

### W&B plots for Task 2.1

Create native W&B line plots from the logged runs, not screenshots or externally generated images:

- Training loss overlay: x-axis `epoch`, y-axis `train/loss`, runs `task2_1_noam` and `task2_1_fixed_lr_1e-4`.
- Validation accuracy overlay: x-axis `epoch`, y-axis `val/token_accuracy`, runs `task2_1_noam` and `task2_1_fixed_lr_1e-4`.
- Optional LR schedule check: x-axis `epoch`, y-axis `optim/lr`.

## Task 2.2: Ablation of the Attention Scaling Factor

Task 2.2 compares scaled dot-product attention against unscaled dot-product attention. The switch is controlled by:

- `--attention_scaling scaled`: uses `QK^T / sqrt(d_k)`.
- `--attention_scaling unscaled`: uses raw `QK^T` without division by `sqrt(d_k)`.

The training script can log Query and Key gradient norms for the first N optimization steps using `--grad_log_steps`. For the assignment, set `--grad_log_steps 1000`.

Use the same scheduler, architecture, batch size, and random seed for both runs. The only intended difference is `--attention_scaling`.

Run the scaled-attention baseline:

```python
!python train.py \
  --project da6401-a3 \
  --task task2.2 \
  --run_name task2_2_scaled_attention \
  --scheduler noam \
  --attention_scaling scaled \
  --grad_log_steps 1000 \
  --epochs 20 \
  --batch_size 64 \
  --d_model 256 \
  --layers 4 \
  --heads 8 \
  --d_ff 1024 \
  --dropout 0.1 \
  --warmup_steps 4000 \
  --lr 1.0 \
  --seed 42
```

Run the unscaled-attention ablation:

```python
!python train.py \
  --project da6401-a3 \
  --task task2.2 \
  --run_name task2_2_unscaled_attention \
  --scheduler noam \
  --attention_scaling unscaled \
  --grad_log_steps 1000 \
  --epochs 20 \
  --batch_size 64 \
  --d_model 256 \
  --layers 4 \
  --heads 8 \
  --d_ff 1024 \
  --dropout 0.1 \
  --warmup_steps 4000 \
  --lr 1.0 \
  --seed 42
```

### W&B plots for Task 2.2

Use native W&B line plots from the two actual runs. Do not upload static screenshots generated outside W&B.

Required gradient-norm plots for the first 1,000 steps:

- Query gradient norm overlay: x-axis `train/global_step`, y-axis `grad_norm/query/all_attention`, runs `task2_2_scaled_attention` and `task2_2_unscaled_attention`.
- Key gradient norm overlay: x-axis `train/global_step`, y-axis `grad_norm/key/all_attention`, runs `task2_2_scaled_attention` and `task2_2_unscaled_attention`.

Useful diagnostic plots:

- First encoder self-attention Query norm: x-axis `train/global_step`, y-axis `grad_norm/query/encoder/layers/0/self_attn`.
- First encoder self-attention Key norm: x-axis `train/global_step`, y-axis `grad_norm/key/encoder/layers/0/self_attn`.
- Training loss overlay: x-axis `epoch`, y-axis `train/loss`.
- Validation accuracy overlay: x-axis `epoch`, y-axis `val/token_accuracy`.
- Learning rate sanity check: x-axis `epoch`, y-axis `optim/lr`; both Task 2.2 runs should match if only scaling differs.

### Task 2.2 analysis checklist

In the W&B report, relate the plots to Section 3.2.1 of "Attention Is All You Need":

- Without the `1/sqrt(d_k)` scaling, dot products tend to have larger magnitudes when `d_k` is large.
- Large attention logits push the softmax toward saturated, near one-hot distributions.
- Saturated softmax outputs have very small local gradients, so the Query and Key projection gradients can become smaller or less stable.
- The scaled run should show healthier Query/Key gradient flow and more stable optimization, while the unscaled run may show weaker, noisier, or less useful Query/Key gradients during the first 1,000 steps.
