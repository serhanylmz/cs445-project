# Stance Detection with Topic-Grouped Attention Networks

This project implements a stance detection system using Topic-Grouped Attention Networks (TGANet) as described in the paper "Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations" by Allaway & McKeown (2020).

## Project Structure

```
.
├── README.md
├── requirements.txt
├── project.py           # Main script
├── data_processing.py   # Data loading and preprocessing
├── model.py            # Model architectures
└── train_eval.py       # Training and evaluation utilities
```

## Features

- Implementation of TGANet for stance detection
- BERT-based baseline model for comparison
- Comprehensive evaluation metrics:
  - Precision, Recall, F1-score
  - Confusion matrices
  - Precision-Recall curves
- TensorBoard integration for training monitoring
- Model checkpointing
- Support for both GPU and CPU training

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the TGANet model:

```bash
python project.py \
    --train_path path/to/train.csv \
    --test_path path/to/test.csv \
    --model_type tganet \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --save_dir checkpoints
```

To train the baseline BERT model:

```bash
python project.py \
    --train_path path/to/train.csv \
    --test_path path/to/test.csv \
    --model_type baseline \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --save_dir checkpoints_baseline
```

### Arguments

- `--train_path`: Path to training data CSV
- `--test_path`: Path to test data CSV
- `--val_size`: Validation set size (default: 0.15)
- `--model_type`: Model type to use ['tganet', 'baseline']
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay for AdamW (default: 0.01)
- `--warmup_steps`: Number of warmup steps (default: 500)
- `--seed`: Random seed (default: 42)
- `--save_dir`: Directory to save model checkpoints and results

## Data Format

The input CSV files should contain the following columns:
- `text`: The text to classify
- `topic`: The topic for stance detection
- `stance`: The stance label ('agree', 'disagree', 'neutral')

## Model Architecture

### TGANet

The TGANet model consists of:
1. BERT encoder for text and topic representation
2. Topic-Grouped Attention mechanism
3. Classification head

Key features:
- Multi-head attention for topic-specific feature extraction
- Scaled dot-product attention following Vaswani et al. (2017)
- Dynamic weighting of text components based on topic relevance

### Baseline BERT

A simple BERT-based classifier that:
1. Encodes text and topic using BERT
2. Uses the [CLS] token representation
3. Applies a classification head

## Evaluation

The system provides comprehensive evaluation metrics:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Precision-Recall curves for each class
- TensorBoard logs for training monitoring

Results are saved in the specified `save_dir`:
- `best_model.pt`: Best model checkpoint
- `confusion_matrix.png`: Confusion matrix visualization
- `pr_curves.png`: Precision-Recall curves
- `logs/`: TensorBoard log directory

## TensorBoard Monitoring

To monitor training progress:

```bash
tensorboard --logdir checkpoints/logs
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{allaway-mckeown-2020-zero,
    title = "Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations",
    author = "Allaway, Emily and McKeown, Kathleen",
    booktitle = "Proceedings of EMNLP 2020",
    year = "2020"
}
``` 