from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Dict, Any, Optional
import random

# Load environment variables
load_dotenv()

class StanceResponse(BaseModel):
    """Structured output for stance detection."""
    stance: str  # One of: "agree", "disagree", "neutral"
    confidence: float  # Between 0 and 1
    explanation: Optional[str] = None  # Optional explanation for the stance

class LLMBaseline:
    def __init__(self, model="gpt-4o-2024-08-06", save_explanations=False, random_seed=42):
        """Initialize the LLM baseline model.
        
        Args:
            model (str): The OpenAI model to use
            save_explanations (bool): Whether to save the model's explanations
            random_seed (int): Random seed for reproducibility
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.save_explanations = save_explanations
        self.explanations = [] if save_explanations else None
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # System prompt for stance detection
        self.system_prompt = """You are an expert at stance detection. Given a text and its topic, determine whether the text expresses agreement, disagreement, or neutrality towards the topic.
        
Rules:
1. Only respond with the structured output format.
2. The stance must be one of: "agree", "disagree", or "neutral"
3. Confidence should be between 0 and 1
4. Be as objective as possible in your assessment
5. Consider both explicit and implicit indicators of stance
6. If the stance is ambiguous, lean towards "neutral" with lower confidence"""

    def predict_single(self, text: str, topic: str) -> Dict[str, Any]:
        """Predict stance for a single text using the LLM."""
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Text: {text}\nTopic: {topic}\nDetermine the stance of the text towards the topic."}
                ],
                response_format=StanceResponse,
            )
            
            response = completion.choices[0].message.parsed
            
            # Map stance to numerical label (matching your existing implementation)
            stance_map = {"agree": 0, "disagree": 1, "neutral": 2}
            label = stance_map[response.stance.lower()]
            
            if self.save_explanations:
                self.explanations.append(response.explanation)
            
            return {
                "label": label,
                "confidence": response.confidence,
                "explanation": response.explanation if self.save_explanations else None
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return neutral with low confidence in case of error
            return {"label": 2, "confidence": 0.33, "explanation": None}

    def sample_test_data(self, test_loader, n_samples=100) -> tuple:
        """Sample a fixed number of examples from the test set.
        
        Args:
            test_loader: DataLoader containing test examples
            n_samples (int): Number of samples to evaluate
            
        Returns:
            tuple: (sampled_data, sampled_labels)
        """
        all_texts = []
        all_topics = []
        all_labels = []
        
        # Collect all test data
        for batch in test_loader:
            all_texts.extend(batch['texts'])
            all_topics.extend(batch['topics'])
            all_labels.extend(batch['labels'].numpy())
        
        # Convert to numpy arrays for easier indexing
        all_texts = np.array(all_texts)
        all_topics = np.array(all_topics)
        all_labels = np.array(all_labels)
        
        # Sample indices
        n_total = len(all_labels)
        indices = random.sample(range(n_total), min(n_samples, n_total))
        
        return (
            all_texts[indices],
            all_topics[indices],
            all_labels[indices]
        )

    def evaluate(self, test_loader, n_samples=100) -> Dict[str, Any]:
        """Evaluate the model on a sampled subset of the test dataset.
        
        Args:
            test_loader: DataLoader containing test examples
            n_samples (int): Number of samples to evaluate
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Sample test data
        sampled_texts, sampled_topics, sampled_labels = self.sample_test_data(
            test_loader, n_samples
        )
        
        # Get predictions
        predictions = []
        confidences = []
        
        for text, topic in tqdm(zip(sampled_texts, sampled_topics), desc="LLM Prediction", total=len(sampled_texts)):
            pred = self.predict_single(text, topic)
            predictions.append(pred["label"])
            confidences.append(pred["confidence"])
        
        predictions = np.array(predictions)
        
        # Calculate metrics (matching your existing implementation)
        metrics = classification_report(sampled_labels, predictions, output_dict=True, zero_division=0)
        metrics['accuracy'] = accuracy_score(sampled_labels, predictions)
        metrics['confusion_matrix'] = confusion_matrix(sampled_labels, predictions)
        
        # Add confidence scores
        metrics['mean_confidence'] = np.mean(confidences)
        metrics['confidence_by_class'] = {
            label: np.mean([conf for pred, conf in zip(predictions, confidences) if pred == label])
            for label in range(3)
        }
        
        if self.save_explanations:
            metrics['explanations'] = self.explanations
        
        return metrics

def main():
    """Example usage of the LLM baseline."""
    from data_processing import load_and_preprocess_data, create_data_loaders
    from baselines import BaselineTracker
    
    # Initialize the tracker
    tracker = BaselineTracker(save_dir='results')
    
    # Load your data
    train_data, val_data, test_data = load_and_preprocess_data(
        train_path='data/train.csv',
        test_path='data/test.csv',
        val_size=0.15
    )
    
    # Create data loaders (using SimpleDataset since we don't need tokenization)
    _, _, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        tokenizer=None,  # Not needed for LLM baseline
        batch_size=32
    )
    
    # Initialize and evaluate LLM baseline
    print("\nRunning LLM Baseline...")
    llm_baseline = LLMBaseline(save_explanations=True, random_seed=42)
    metrics = llm_baseline.evaluate(test_loader, n_samples=100)
    
    # Add metrics to tracker
    tracker.add_metrics('LLM', metrics)
    
    # Generate plots and save results
    tracker.plot_comparison_bars()
    tracker.plot_confusion_matrices()
    tracker.save_results()
    
    # Print results
    print("\nLLM Baseline Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro avg']['f1-score']:.4f}")
    print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    print("\nConfidence by class:")
    for label, conf in metrics['confidence_by_class'].items():
        stance = ['agree', 'disagree', 'neutral'][label]
        print(f"{stance}: {conf:.4f}")

if __name__ == "__main__":
    main() 