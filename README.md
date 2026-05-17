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

## Links

- **GitHub Repository:** https://github.com/Abdallahdula/da6401_assignment_3_abdallah/tree/main
- **W&B Report:** https://api.wandb.ai/links/zda23m016-iit-madras-zanzibar/x02ekt0k
