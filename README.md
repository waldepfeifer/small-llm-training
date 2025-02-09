<img width="1020" alt="image" src="https://github.com/user-attachments/assets/a8c0a31e-30ed-4a95-a2a9-082efb9319df" />

## Overview
This project provides a simple GPT-style language model for local execution. You can train it on any text dataset and generate context-relevant outputs based on prompts. While the provided example uses the Bible, any text source can be used.

## Key Features
- Train on custom text datasets.
- Generate text using user prompts.
- Export the trained model for reuse.
- Lightweight design for local machines.

## Setup
### Requirements
- `torch`
- `numpy`

Install with:
```bash
pip install torch numpy
```

### Dataset
1. Save your text dataset as input.txt.
2. Ensure the file is in the project directory.

## Running the Model

### Train the Model

Run this command to train:
```bash
python train_gpt.py
```
Training will display loss metrics during execution.

### Generate Text

To generate text with a prompt:
```bash
python train_gpt.py "Your prompt here"
```
Example:
```bash
python train_gpt.py "The birth of moses"
```
### Save Model

The trained model is saved as llm_model.pth automatically.

## Hyperparameter Tuning

You can modify these parameters directly in the script to control performance:

- batch_size: Number of sequences processed in parallel.
- block_size: Maximum sequence length.
- n_embd: Embedding dimension (controls model size).
- n_head: Number of attention heads.
- n_layer: Number of transformer layers.
- dropout: Dropout probability to prevent overfitting.
- learning_rate: Adjust for faster or more stable training.

### Example Adjustments:
batch_size = 32  # For larger batch processing
n_layer = 4      # Use fewer layers to save memory
learning_rate = 1e-4  # Lower learning rate for fine-tuning

## Tips

For faster experiments, use a small dataset.
Lower n_layer and n_head if running on a low-resource machine.
This model offers a simple and flexible way to explore text generation on your local computer.

