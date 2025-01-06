import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pandas as pd
import os
import logging
from datetime import datetime
from model import TGANet
from baselines import (
    BaselineTracker, MajorityClassBaseline, RandomBaseline, TfidfBaseline,
    classification_report, confusion_matrix, accuracy_score
)
from data_processing import load_and_preprocess_data, create_data_loaders
from transformers import (
    BertTokenizer, AdamW, get_linear_schedule_with_warmup
)
import json
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentTracker(BaselineTracker):
    def __init__(self, save_dir='results'):
        super().__init__(save_dir)
        self.writer = SummaryWriter(os.path.join(self.results_dir, 'tensorboard'))
        self.training_losses = {}
        self.validation_losses = {}
        
    def add_loss(self, model_name, epoch, train_loss, val_loss=None):
        if model_name not in self.training_losses:
            self.training_losses[model_name] = []
            self.validation_losses[model_name] = []
            
        self.training_losses[model_name].append(train_loss)
        if val_loss is not None:
            self.validation_losses[model_name].append(val_loss)
            
        self.writer.add_scalar(f'{model_name}/train_loss', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
            
    def plot_topic_attention_weights(self, attention_weights, topics, save_name):
        plt.figure(figsize=(15, 10))
        sns.heatmap(attention_weights, xticklabels=topics, yticklabels=topics, 
                    cmap='viridis')
        plt.title('Topic Attention Weights')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{save_name}.png'))
        plt.close()

def evaluate_model(model, eval_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels=labels)
            preds = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(eval_loader), metrics

def train_tganet(model, train_loader, val_loader, test_loader, 
                 device, tracker, args):
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Calculate total steps for scheduler
    total_steps = len(train_loader) * args.num_epochs
    
    # Initialize scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _ = model(input_ids, attention_mask, labels=labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, device)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logger.info(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        
        tracker.add_loss('TGANet', epoch, train_loss, val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(tracker.results_dir, 'tganet_best.pt'))
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(tracker.results_dir, 'tganet_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    _, test_metrics = evaluate_model(model, test_loader, device)
    tracker.add_metrics('TGANet', test_metrics)
    
    return model

def run_baselines(train_loader, val_loader, test_loader, tracker):
    # 1. Random Baseline (no training needed, evaluate on test)
    print("\nRunning Random Baseline...")
    random_baseline = RandomBaseline()
    preds, labels = random_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('Random', metrics)
    
    # 2. Majority Class Baseline (fit on train, evaluate on test)
    print("\nTraining Majority Class Baseline...")
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(train_loader)
    preds, labels = majority_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('MajorityClass', metrics)
    
    # 3. Naive Bayes + TF-IDF (with hyperparameter tuning)
    print("\nTraining Naive Bayes + TF-IDF...")
    nb_baseline = TfidfBaseline(
        MultinomialNB(),
        "Naive Bayes + TF-IDF"
    )
    nb_baseline.fit(train_loader, val_loader)  # Use validation set for hyperparameter tuning
    preds, labels = nb_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('NaiveBayes', metrics)
    
    # 4. Random Forest + TF-IDF (with hyperparameter tuning)
    print("\nTraining Random Forest + TF-IDF...")
    rf_baseline = TfidfBaseline(
        RandomForestClassifier(n_jobs=-1),
        "Random Forest + TF-IDF"
    )
    rf_baseline.fit(train_loader, val_loader)  # Use validation set for hyperparameter tuning
    preds, labels = rf_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('RandomForest', metrics)
    
    # 5. Logistic Regression + TF-IDF (with hyperparameter tuning)
    print("\nTraining Logistic Regression + TF-IDF...")
    lr_baseline = TfidfBaseline(
        LogisticRegression(n_jobs=-1),
        "Logistic Regression + TF-IDF"
    )
    lr_baseline.fit(train_loader, val_loader)  # Use validation set for hyperparameter tuning
    preds, labels = lr_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('LogisticRegression', metrics)

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = ExperimentTracker(save_dir=args.save_dir)
    
    logger.info(f'Using device: {device}')
    logger.info(f'Results will be saved in: {tracker.results_dir}')
    
    # Load data for baselines (without tokenizer)
    train_data, val_data, test_data = load_and_preprocess_data(
        args.train_path, args.test_path, val_size=args.val_size
    )
    
    # Create data loaders for baselines
    if not args.skip_baselines:
        logger.info('Running baselines...')
        baseline_train_loader, baseline_val_loader, baseline_test_loader = create_data_loaders(
            train_data, val_data, test_data, None, batch_size=args.batch_size
        )
        run_baselines(baseline_train_loader, baseline_val_loader, baseline_test_loader, tracker)
    
    # Create tokenized data loaders for TGANet
    logger.info('Preparing data for TGANet...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, tokenizer, batch_size=args.batch_size
    )
    
    # Train TGANet
    logger.info('Training TGANet...')
    tganet = TGANet(
        num_labels=3,
        hidden_dropout_prob=args.dropout,
        topic_clusters=args.topic_clusters
    ).to(device)
    
    trained_model = train_tganet(
        tganet, 
        train_loader, 
        val_loader, 
        test_loader,
        device, 
        tracker,
        args
    )
    
    # Generate final plots and save results
    tracker.plot_comparison_bars()
    tracker.plot_confusion_matrices()
    
    # Save hyperparameters and configuration
    config = vars(args)
    config['device'] = str(device)
    with open(os.path.join(tracker.results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f'\nExperiment completed! Results saved in: {tracker.results_dir}')
    logger.info('\nFinal Test Metrics:')
    for model_name, metrics in tracker.test_metrics.items():
        logger.info(f'\n{model_name}:')
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro avg']['f1-score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate TGANet with baselines')
    
    # Data parameters
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Validation set size')
    
    # Model parameters
    parser.add_argument('--topic_clusters', type=int, default=50,
                        help='Number of topic clusters')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip running baselines')
    
    args = parser.parse_args()
    main(args) 