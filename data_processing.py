import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import re

def load_data(file_path):
    """
    Load data file with proper handling of delimiters and encoding
    """
    try:
        # First attempt - try comma delimiter with latin1 encoding
        df = pd.read_csv(
            file_path,
            encoding='latin1',
            engine='python',
            on_bad_lines='skip'
        )

        # Check if we got one column with everything
        if len(df.columns) == 1:
            print("File appears to be tab-delimited but was read as CSV. Trying again with tabs...")

            # Try again with tab delimiter
            df = pd.read_csv(
                file_path,
                sep='\t',
                encoding='latin1',
                engine='python',
                on_bad_lines='skip'
            )

        print(f"Successfully loaded {len(df)} rows from {file_path}")

        # Clean up column names (remove any extra quotes or spaces)
        df.columns = [col.strip().strip('"').strip("'") for col in df.columns]

        # Verify expected columns
        expected_columns = ['Tweet', 'Target', 'Stance', 'Opinion Towards', 'Sentiment']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")

        return df

    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

        # Try one more time with direct file reading
        try:
            print("\nAttempting to read file directly...")
            with open(file_path, 'r', encoding='latin1') as file:
                lines = file.readlines()

            # Process headers
            headers = lines[0].strip().split('\t')
            headers = [h.strip().strip('"').strip("'") for h in headers]

            # Process data
            data = []
            for line in lines[1:]:
                values = line.strip().split('\t')
                if len(values) == len(headers):
                    row = dict(zip(headers, values))
                    data.append(row)

            df = pd.DataFrame(data)
            print(f"Successfully loaded {len(df)} rows using direct file reading")
            return df

        except Exception as e2:
            print(f"Direct file reading failed: {str(e2)}")
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
    print("Loading training data...")
    train_df = load_data(train_path)
    print("Loading test data...")
    test_df = load_data(test_path)

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