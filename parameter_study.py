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
from data_processing import load_and_preprocess_data, create_data_loaders
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import json
import itertools
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParameterStudyTracker:
    def __init__(self, base_dir='parameter_study_results'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(base_dir, f'study_{self.timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.all_results = {
            'configs': [],
            'metrics': [],
            'training_curves': {}
        }
    
    def add_run_results(self, config, metrics, training_curve):
        """Store results from a single run"""
        run_id = len(self.all_results['configs'])
        
        # Store configuration
        config_copy = config.copy()
        config_copy['run_id'] = run_id
        self.all_results['configs'].append(config_copy)
        
        # Store metrics
        metrics_copy = metrics.copy()
        metrics_copy['run_id'] = run_id
        self.all_results['metrics'].append(metrics_copy)
        
        # Store training curve
        self.all_results['training_curves'][run_id] = training_curve
    
    def plot_training_curves(self):
        """Plot training curves for all runs"""
        plt.figure(figsize=(15, 10))
        
        for run_id, curve in self.all_results['training_curves'].items():
            config = self.all_results['configs'][run_id]
            label = f"bs={config['batch_size']}, lr={config['learning_rate']}, dp={config['dropout']}"
            
            steps = range(len(curve['train_loss']))
            plt.plot(steps, curve['train_loss'], label=f"{label} (train)")
            plt.plot(steps, curve['val_loss'], label=f"{label} (val)", linestyle='--')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Curves for All Configurations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), bbox_inches='tight')
        plt.close()
    
    def plot_parameter_comparison(self):
        """Create bar plots comparing different parameter settings"""
        results_df = pd.DataFrame(self.all_results['metrics'])
        configs_df = pd.DataFrame(self.all_results['configs'])
        df = pd.concat([configs_df, results_df], axis=1)
        
        metrics = ['accuracy', 'macro_f1']
        params = ['batch_size', 'learning_rate', 'dropout']
        
        fig, axes = plt.subplots(len(params), len(metrics), figsize=(15, 20))
        
        for i, param in enumerate(params):
            for j, metric in enumerate(metrics):
                sns.barplot(data=df, x=param, y=metric, ax=axes[i,j])
                axes[i,j].set_title(f'{metric} vs {param}')
                axes[i,j].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameter_comparison.png'))
        plt.close()
    
    def save_results(self):
        """Save all results to JSON"""
        results = {
            'configs': self.all_results['configs'],
            'metrics': self.all_results['metrics'],
            'training_curves': {
                str(k): v for k, v in self.all_results['training_curves'].items()
            }
        }
        
        with open(os.path.join(self.results_dir, 'study_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Also save as CSV for easy analysis
        pd.DataFrame(self.all_results['metrics']).to_csv(
            os.path.join(self.results_dir, 'metrics.csv'), index=False
        )
        pd.DataFrame(self.all_results['configs']).to_csv(
            os.path.join(self.results_dir, 'configs.csv'), index=False
        )

def evaluate_model(model, eval_loader, device):
    """Evaluate model on given loader"""
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
    
    return {
        'loss': total_loss / len(eval_loader),
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro avg']['f1-score']
    }

def train_one_epoch(model, train_loader, val_loader, optimizer, scheduler, device):
    """Train for one epoch and track metrics"""
    model.train()
    total_loss = 0
    training_curve = {
        'train_loss': [],
        'val_loss': []
    }
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss, _ = model(input_ids, attention_mask, labels=labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        training_curve['train_loss'].append(loss.item())
        
        # Evaluate every 100 steps
        if len(training_curve['train_loss']) % 100 == 0:
            val_metrics = evaluate_model(model, val_loader, device)
            training_curve['val_loss'].append(val_metrics['loss'])
            
            progress_bar.set_postfix({
                'train_loss': loss.item(),
                'val_loss': val_metrics['loss']
            })
    
    return training_curve

def run_configuration(config, train_loader, val_loader, test_loader, device):
    """Run training with a specific configuration"""
    logger.info(f"\nRunning configuration: {config}")
    
    # Initialize model
    model = TGANet(
        num_labels=3,
        hidden_dropout_prob=config['dropout'],
        topic_clusters=50  # Fixed for this study
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01  # Fixed for this study
    )
    
    total_steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,  # Fixed for this study
        num_training_steps=total_steps
    )
    
    # Train for one epoch
    training_curve = train_one_epoch(
        model, train_loader, val_loader, optimizer, scheduler, device
    )
    
    # Final evaluation
    test_metrics = evaluate_model(model, test_loader, device)
    
    return test_metrics, training_curve

def main():
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize tracker
    tracker = ParameterStudyTracker()
    logger.info(f'Results will be saved in: {tracker.results_dir}')
    
    # Define parameter grid
    param_grid = {
        'batch_size': [16, 32],
        'learning_rate': [1e-4, 2e-5],
        'dropout': [0.0, 0.2]
    }
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load data once
    train_data, val_data, test_data = load_and_preprocess_data(
        'data/train.csv',
        'data/test.csv',
        val_size=0.15
    )
    
    # Generate all parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) 
        for v in itertools.product(*param_grid.values())
    ]
    
    # Run each configuration
    for config in param_combinations:
        # Create data loaders with current batch size
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data,
            tokenizer, batch_size=config['batch_size']
        )
        
        # Run training and evaluation
        metrics, training_curve = run_configuration(
            config, train_loader, val_loader, test_loader, device
        )
        
        # Store results
        tracker.add_run_results(config, metrics, training_curve)
        
        logger.info(f"Results for configuration {config}:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # Generate plots and save results
    tracker.plot_training_curves()
    tracker.plot_parameter_comparison()
    tracker.save_results()
    
    logger.info(f"\nParameter study completed! Results saved in: {tracker.results_dir}")

if __name__ == "__main__":
    main() 