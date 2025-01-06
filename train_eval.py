import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        device,
        num_epochs=10,
        scheduler=None,
        save_dir='checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        
        self.best_val_f1 = 0.0
        self.best_model_path = os.path.join(save_dir, 'best_model.pt')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('F1-macro/train', report['macro avg']['f1-score'], epoch)
        
        return avg_loss, report

    def evaluate(self, dataloader, mode='val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Evaluating {mode}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        report = classification_report(all_labels, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Calculate precision-recall curves
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # One-vs-Rest precision-recall curves
        pr_curves = {}
        for i in range(3):  # 3 classes
            precision, recall, _ = precision_recall_curve(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            ap = average_precision_score(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            pr_curves[i] = {
                'precision': precision,
                'recall': recall,
                'ap': ap
            }
        
        return avg_loss, report, conf_matrix, pr_curves

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            
            # Training
            train_loss, train_report = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_report, val_conf_matrix, val_pr_curves = self.evaluate(
                self.val_loader,
                mode='val'
            )
            
            # Log metrics
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('F1-macro/val', val_report['macro avg']['f1-score'], epoch)
            
            # Save best model
            if val_report['macro avg']['f1-score'] > self.best_val_f1:
                self.best_val_f1 = val_report['macro avg']['f1-score']
                # Save full checkpoint as .tar
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_f1': self.best_val_f1,
                    'loss': val_loss
                }, os.path.join(self.save_dir, 'best_model.tar'))
                # Also save just the model state dict as .pt for compatibility
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f'New best model saved with F1: {self.best_val_f1:.4f}')
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print('\nValidation Report:')
            print(f"{'':>12} precision  recall  f1-score  support")
            print(f"{'-'*50}")
            for label, metrics in val_report.items():
                if label in ['agree', 'disagree', 'neutral']:
                    print(f"{label:>12} {metrics['precision']:9.2f} {metrics['recall']:7.2f} {metrics['f1-score']:9.2f} {metrics['support']:8.0f}")
            print(f"{'-'*50}")
            print(f"{'macro avg':>12} {val_report['macro avg']['precision']:9.2f} {val_report['macro avg']['recall']:7.2f} {val_report['macro avg']['f1-score']:9.2f} {val_report['macro avg']['support']:8.0f}")
            print(f"{'weighted avg':>12} {val_report['weighted avg']['precision']:9.2f} {val_report['weighted avg']['recall']:7.2f} {val_report['weighted avg']['f1-score']:9.2f} {val_report['weighted avg']['support']:8.0f}")
        
        self.writer.close()
        return self.best_model_path

    def plot_confusion_matrix(self, conf_matrix, title='Confusion Matrix'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['agree', 'disagree', 'neutral'],
            yticklabels=['agree', 'disagree', 'neutral']
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()

    def plot_pr_curves(self, pr_curves, title='Precision-Recall Curves'):
        plt.figure(figsize=(10, 8))
        classes = ['agree', 'disagree', 'neutral']
        for i, label in enumerate(classes):
            plt.plot(
                pr_curves[i]['recall'],
                pr_curves[i]['precision'],
                label=f'{label} (AP = {pr_curves[i]["ap"]:.2f})'
            )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        return plt.gcf()

def evaluate_model(trainer, model_path):
    """Evaluate a trained model and generate comprehensive reports."""
    # Load best model
    model = trainer.model
    if model_path.endswith('.tar'):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # .pt file
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Evaluate on test set
    test_loss, test_report, test_conf_matrix, test_pr_curves = trainer.evaluate(
        trainer.test_loader,
        mode='test'
    )
    
    # Save confusion matrix plot
    conf_matrix_fig = trainer.plot_confusion_matrix(
        test_conf_matrix,
        title='Test Set Confusion Matrix'
    )
    conf_matrix_fig.savefig(os.path.join(trainer.save_dir, 'confusion_matrix.png'))
    
    # Save PR curves plot
    pr_curves_fig = trainer.plot_pr_curves(
        test_pr_curves,
        title='Test Set Precision-Recall Curves'
    )
    pr_curves_fig.savefig(os.path.join(trainer.save_dir, 'pr_curves.png'))
    
    # Print test results
    print('\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print('\nClassification Report:')
    print(f"{'':>12} precision  recall  f1-score  support")
    print(f"{'-'*50}")
    for label, metrics in test_report.items():
        if label in ['agree', 'disagree', 'neutral']:
            print(f"{label:>12} {metrics['precision']:9.2f} {metrics['recall']:7.2f} {metrics['f1-score']:9.2f} {metrics['support']:8.0f}")
    print(f"{'-'*50}")
    print(f"{'macro avg':>12} {test_report['macro avg']['precision']:9.2f} {test_report['macro avg']['recall']:7.2f} {test_report['macro avg']['f1-score']:9.2f} {test_report['macro avg']['support']:8.0f}")
    print(f"{'weighted avg':>12} {test_report['weighted avg']['precision']:9.2f} {test_report['weighted avg']['recall']:7.2f} {test_report['weighted avg']['f1-score']:9.2f} {test_report['weighted avg']['support']:8.0f}")
    
    return test_report, test_conf_matrix, test_pr_curves 