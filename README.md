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

## Task 2.3: Attention Rollout & Head Specialization

Task 2.3 logs the attention weights from the **last encoder layer** for one validation source sentence. In this German-to-English setup, the encoder input is the German source sentence from Multi30k. If your report wording says "English sentence," explain that this repository's encoder side is German; the same code applies to English encoder tokens if you reverse or replace the dataset.

The implementation captures the attention matrix inside every `MultiHeadAttention` module during the forward pass. Passing `--log_attention_maps` runs one final encoder pass after training and logs one heat map per individual head from `model.encoder.layers[-1].self_attn`.

Run a scaled-attention model and log the last-encoder attention maps at the end of training:

```python
!python train.py \
  --project da6401-a3 \
  --task task2.3 \
  --run_name task2_3_last_encoder_heads \
  --scheduler noam \
  --attention_scaling scaled \
  --log_attention_maps \
  --attention_sample_index 0 \
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

### W&B panels for Task 2.3

Use the W&B artifacts logged by the run itself. Do not paste external screenshots into the report.

Required heat maps:

- `attention/last_encoder/head_0_heatmap`
- `attention/last_encoder/head_1_heatmap`
- Continue through `attention/last_encoder/head_7_heatmap` when `--heads 8` is used.

Raw attention-weight tables are also logged for reproducibility:

- `attention/last_encoder/head_0_weights`
- `attention/last_encoder/head_1_weights`
- Continue through all heads.

Useful summary table and scalar diagnostics:

- `attention/last_encoder/head_statistics` summarizes each head's diagonal attention, next-token attention, previous-token attention, `<eos>` attention, average attention distance, and entropy.
- `attention/last_encoder/head_<n>_diagonal_mean` can indicate a head that mostly copies or focuses on the same token position.
- `attention/last_encoder/head_<n>_next_token_mean` can indicate a head that frequently attends to the next token.
- `attention/last_encoder/head_<n>_avg_attention_distance` can help identify heads that capture longer-range dependencies.
- `attention/last_encoder/head_<n>_entropy` can help compare sharp, specialized heads against diffuse heads.

### Task 2.3 analysis checklist

In the W&B report, inspect each logged head heat map and the `head_statistics` table:

- Identify at least one head with a strong diagonal pattern if present; this suggests same-position/local-token tracking.
- Identify at least one head with high next-token or previous-token attention if present; this suggests local sequential structure.
- Identify at least one head with high average attention distance or clear off-diagonal bands if present; this suggests longer-range dependency capture.
- Discuss head redundancy by comparing heat maps. If multiple heads show very similar diffuse, diagonal, or `<eos>`-focused patterns, state that they appear redundant. If heads show visibly different patterns and different statistics, state that they appear specialized.
- Include the exact source tokens from `attention/last_encoder/source_tokens` so readers know which sentence produced the heat maps.

## Task 2.4: Positional Encoding vs Learned Positional Embeddings

This repository now supports both positional strategies through `--positional_encoding`:

- `sinusoidal`: fixed sinusoidal encoding (default).
- `learned`: trainable positional embedding via `torch.nn.Embedding`.

Run the sinusoidal baseline:

```python
!python train.py \
  --project da6401-a3 \
  --task task2.4 \
  --run_name task2_4_sinusoidal \
  --scheduler noam \
  --attention_scaling scaled \
  --positional_encoding sinusoidal \
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

Run the learned positional embedding variant:

```python
!python train.py \
  --project da6401-a3 \
  --task task2.4 \
  --run_name task2_4_learned_positional \
  --scheduler noam \
  --attention_scaling scaled \
  --positional_encoding learned \
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

### What to plot for Task 2.4

Create W&B line plots comparing the two runs:

- **Main plot (required):** x-axis `epoch`, y-axis `val/bleu` (overlay both runs).
- **Support plot:** x-axis `epoch`, y-axis `val/token_accuracy`.
- **Support plot:** x-axis `epoch`, y-axis `val/loss`.

For your report discussion, include the final `val/bleu` from each run and explain whether learned positional embeddings helped in-distribution validation performance.

### Theoretical note to include in your report

Sinusoidal encodings are generated by a deterministic function of absolute position, so positional vectors can be computed for positions beyond the training range without adding new parameters. Learned positional embeddings only have explicit vectors for positions seen/indexed during training; they do not provide the same built-in functional extrapolation to unseen longer sequence lengths.
