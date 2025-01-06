import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_processing import load_and_preprocess_data, create_data_loaders
import random

class BaselineTracker:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(save_dir, f'baseline_results_{self.timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        self.test_metrics = {}
            
    def add_metrics(self, model_name, metrics):
        # Convert numpy arrays to lists for JSON serialization
        processed_metrics = {}
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                processed_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                processed_metrics[key] = {k: float(v) if isinstance(v, np.float32) else v 
                                       for k, v in value.items()}
            elif isinstance(value, np.float32):
                processed_metrics[key] = float(value)
            else:
                processed_metrics[key] = value
                
        self.test_metrics[model_name] = processed_metrics
            
    def plot_comparison_bars(self):
        plt.figure(figsize=(15, 8))
        
        models = list(self.test_metrics.keys())
        metrics = {
            'Accuracy': [self.test_metrics[model]['accuracy'] for model in models],
            'Macro F1': [self.test_metrics[model]['macro avg']['f1-score'] for model in models]
        }
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            plt.bar(x + i*width, values, width, label=metric_name)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width/2, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'))
        plt.close()
        
    def plot_confusion_matrices(self):
        for model_name, metrics in self.test_metrics.items():
            if 'confusion_matrix' in metrics:
                plt.figure(figsize=(10, 8))
                sns.heatmap(metrics['confusion_matrix'], 
                           annot=True, 
                           fmt='d',
                           cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(os.path.join(self.results_dir, f'confusion_matrix_{model_name}.png'))
                plt.close()
                
    def save_results(self):
        results = {
            'test_metrics': self.test_metrics
        }
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

class MajorityClassBaseline:
    def __init__(self):
        self.majority_class = None
        
    def fit(self, loader):
        all_labels = []
        for batch in loader:
            all_labels.extend(batch['labels'].numpy())
        self.majority_class = np.argmax(np.bincount(all_labels))
        print(f"Majority class is: {self.majority_class}")
        
    def predict(self, loader):
        all_preds = []
        all_labels = []
        for batch in loader:
            batch_size = len(batch['labels'])
            all_preds.extend([self.majority_class] * batch_size)
            all_labels.extend(batch['labels'].numpy())
        return all_preds, all_labels

class RandomBaseline:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        
    def predict(self, loader):
        all_preds = []
        all_labels = []
        for batch in loader:
            batch_size = len(batch['labels'])
            all_preds.extend([random.randint(0, self.n_classes-1) for _ in range(batch_size)])
            all_labels.extend(batch['labels'].numpy())
        return all_preds, all_labels

class TfidfBaseline:
    def __init__(self, classifier, name):
        self.text_vectorizer = TfidfVectorizer(max_features=10000)
        self.topic_vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = classifier
        self.name = name
        self.best_model = None
        
    def fit(self, train_loader, val_loader=None):
        # Collect training data
        train_texts = []
        train_topics = []
        train_labels = []
        
        for batch in train_loader:
            train_texts.extend(batch['texts'])
            train_topics.extend(batch['topics'])
            train_labels.extend(batch['labels'].numpy())
            
        # Transform training data
        train_text_features = self.text_vectorizer.fit_transform(train_texts)
        train_topic_features = self.topic_vectorizer.fit_transform(train_topics)
        X_train = np.hstack([train_text_features.toarray(), train_topic_features.toarray()])
        
        print(f"\nTraining {self.name}...")
        print(f"Feature matrix shape: {X_train.shape}")
        
        if val_loader is not None:
            # If we have validation data, use it for hyperparameter tuning
            val_texts = []
            val_topics = []
            val_labels = []
            
            for batch in val_loader:
                val_texts.extend(batch['texts'])
                val_topics.extend(batch['topics'])
                val_labels.extend(batch['labels'].numpy())
            
            # Transform validation data
            val_text_features = self.text_vectorizer.transform(val_texts)
            val_topic_features = self.topic_vectorizer.transform(val_topics)
            X_val = np.hstack([val_text_features.toarray(), val_topic_features.toarray()])
            
            # If classifier supports predict_proba, we can use it for model selection
            if hasattr(self.classifier, 'predict_proba'):
                best_score = -np.inf
                best_params = None
                
                # Example hyperparameter tuning for different classifiers
                if isinstance(self.classifier, MultinomialNB):
                    alphas = [0.1, 0.5, 1.0, 2.0]
                    for alpha in alphas:
                        model = MultinomialNB(alpha=alpha)
                        model.fit(X_train, train_labels)
                        val_score = model.score(X_val, val_labels)
                        if val_score > best_score:
                            best_score = val_score
                            best_params = {'alpha': alpha}
                            self.best_model = model
                
                elif isinstance(self.classifier, RandomForestClassifier):
                    n_estimators_list = [50, 100, 200]
                    max_depths = [None, 10, 20]
                    for n_est in n_estimators_list:
                        for depth in max_depths:
                            model = RandomForestClassifier(
                                n_estimators=n_est, 
                                max_depth=depth,
                                n_jobs=-1
                            )
                            model.fit(X_train, train_labels)
                            val_score = model.score(X_val, val_labels)
                            if val_score > best_score:
                                best_score = val_score
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth
                                }
                                self.best_model = model
                
                elif isinstance(self.classifier, LogisticRegression):
                    C_values = [0.1, 1.0, 10.0]
                    for C in C_values:
                        model = LogisticRegression(
                            C=C,
                            max_iter=1000,
                            n_jobs=-1
                        )
                        model.fit(X_train, train_labels)
                        val_score = model.score(X_val, val_labels)
                        if val_score > best_score:
                            best_score = val_score
                            best_params = {'C': C}
                            self.best_model = model
                
                print(f"Best validation score: {best_score:.4f}")
                print(f"Best parameters: {best_params}")
            
            if self.best_model is None:
                # If no hyperparameter tuning was done, use the original classifier
                self.best_model = self.classifier
                self.best_model.fit(X_train, train_labels)
        else:
            # If no validation data, just fit on training data
            self.best_model = self.classifier
            self.best_model.fit(X_train, train_labels)
        
    def predict(self, loader):
        texts = []
        topics = []
        labels = []
        
        for batch in loader:
            texts.extend(batch['texts'])
            topics.extend(batch['topics'])
            labels.extend(batch['labels'].numpy())
            
        # Transform using fitted vectorizers
        text_features = self.text_vectorizer.transform(texts)
        topic_features = self.topic_vectorizer.transform(topics)
        
        # Concatenate features
        X = np.hstack([text_features.toarray(), topic_features.toarray()])
        
        predictions = self.best_model.predict(X)
        return predictions, labels

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Setup
    tracker = BaselineTracker()
    print(f"Results will be saved in: {tracker.results_dir}")
    
    # Load data
    train_data, val_data, test_data = load_and_preprocess_data(
        'data/train.csv', 'data/test.csv'
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, None
    )
    
    # 1. Random Baseline
    print("\nRunning Random Baseline...")
    random_baseline = RandomBaseline()
    preds, labels = random_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('Random', metrics)
    
    # 2. Majority Class Baseline
    print("\nTraining Majority Class Baseline...")
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(train_loader)
    preds, labels = majority_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('MajorityClass', metrics)
    
    # 3. Naive Bayes + TF-IDF (from milestone)
    nb_baseline = TfidfBaseline(
        MultinomialNB(),
        "Naive Bayes + TF-IDF"
    )
    nb_baseline.fit(train_loader)
    preds, labels = nb_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('NaiveBayes', metrics)
    
    # 4. Random Forest + TF-IDF
    rf_baseline = TfidfBaseline(
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Random Forest + TF-IDF"
    )
    rf_baseline.fit(train_loader)
    preds, labels = rf_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('RandomForest', metrics)
    
    # 5. Logistic Regression + TF-IDF
    lr_baseline = TfidfBaseline(
        LogisticRegression(max_iter=1000, n_jobs=-1),
        "Logistic Regression + TF-IDF"
    )
    lr_baseline.fit(train_loader)
    preds, labels = lr_baseline.predict(test_loader)
    metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    tracker.add_metrics('LogisticRegression', metrics)
    
    # Generate plots and save results
    tracker.plot_comparison_bars()
    tracker.plot_confusion_matrices()
    tracker.save_results()
    
    print("\nFinal Results:")
    for model_name, metrics in tracker.test_metrics.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro avg']['f1-score']:.4f}")
    print(f"\nResults saved in: {tracker.results_dir}")

if __name__ == "__main__":
    main() 