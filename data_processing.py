import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

class VASTDataset(Dataset):
    def __init__(self, texts, topics, labels, tokenizer, max_length=512):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        topic = str(self.topics[idx])
        label = self.labels[idx]

        # Encode text and topic together with special [SEP] token
        encoding = self.tokenizer.encode_plus(
            text,
            topic,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(train_path, test_path, val_size=0.15, random_state=42):
    """Load and preprocess the VAST dataset."""
    # Load data with different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    def try_read_csv(file_path):
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read file {file_path} with any of the attempted encodings: {encodings}")

    print("Loading training data...")
    train_df = try_read_csv(train_path)
    print("Loading test data...")
    test_df = try_read_csv(test_path)

    # Convert stance labels to numeric
    label_map = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    train_df['label'] = train_df['Stance'].map(label_map)
    test_df['label'] = test_df['Stance'].map(label_map)

    # Split training data into train and validation
    train_data, val_data = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df['label']
    )

    return train_data, val_data, test_df

def create_data_loaders(train_data, val_data, test_data, tokenizer, batch_size=16):
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    train_dataset = VASTDataset(
        texts=train_data['Tweet'].values,
        topics=train_data['Target'].values,
        labels=train_data['label'].values,
        tokenizer=tokenizer
    )

    val_dataset = VASTDataset(
        texts=val_data['Tweet'].values,
        topics=val_data['Target'].values,
        labels=val_data['label'].values,
        tokenizer=tokenizer
    )

    test_dataset = VASTDataset(
        texts=test_data['Tweet'].values,
        topics=test_data['Target'].values,
        labels=test_data['label'].values,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader 