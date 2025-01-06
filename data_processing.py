import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import re

def load_data(file_path):
    """Load VAST dataset with proper encoding"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            print(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags symbol (but keep the text)
    text = re.sub(r'#', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SimpleDataset(Dataset):
    """Dataset class for non-tokenized data (e.g., for majority class baseline)"""
    def __init__(self, texts, topics, labels):
        self.texts = texts
        self.topics = topics
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'topics': self.topics[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class VASTDataset(Dataset):
    def __init__(self, texts, topics, labels, tokenizer, max_length=128):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_text(str(self.texts[idx]))
        topic = preprocess_text(str(self.topics[idx]))
        label = self.labels[idx]

        # Combine text and topic with a separator
        combined_text = f"{text} [SEP] {topic}"
        
        # Encode the combined text as a single sequence
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(train_path, test_path, val_size=0.15, random_state=42):
    """Load and preprocess the VAST dataset."""
    print("Loading training data...")
    train_df = load_data(train_path)
    print("Loading test data...")
    test_df = load_data(test_path)

    # Labels are already numeric in VAST dataset
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    # Split training data into train and validation
    train_data, val_data = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df['label']
    )

    return train_data, val_data, test_df

def create_data_loaders(train_data, val_data, test_data, tokenizer=None, batch_size=16):
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    
    if tokenizer is not None:
        # Use VASTDataset with tokenizer
        train_dataset = VASTDataset(
            texts=train_data['post'].values,
            topics=train_data['new_topic'].values,
            labels=train_data['label'].values,
            tokenizer=tokenizer
        )

        val_dataset = VASTDataset(
            texts=val_data['post'].values,
            topics=val_data['new_topic'].values,
            labels=val_data['label'].values,
            tokenizer=tokenizer
        )

        test_dataset = VASTDataset(
            texts=test_data['post'].values,
            topics=test_data['new_topic'].values,
            labels=test_data['label'].values,
            tokenizer=tokenizer
        )
    else:
        # Use SimpleDataset without tokenizer
        train_dataset = SimpleDataset(
            texts=train_data['post'].values,
            topics=train_data['new_topic'].values,
            labels=train_data['label'].values
        )

        val_dataset = SimpleDataset(
            texts=val_data['post'].values,
            topics=val_data['new_topic'].values,
            labels=val_data['label'].values
        )

        test_dataset = SimpleDataset(
            texts=test_data['post'].values,
            topics=test_data['new_topic'].values,
            labels=test_data['label'].values
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader 