import matplotlib.pyplot as plt
import numpy as np
import json

def plot_comparison_bars(results_file='results.json'):
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    test_metrics = results['test_metrics']
    models = list(test_metrics.keys())
    metrics = {
        'Accuracy': [test_metrics[model]['accuracy'] for model in models],
        'Macro F1': [test_metrics[model]['macro avg']['f1-score'] for model in models]
    }
    
    # Create plot
    plt.figure(figsize=(15, 8))
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
    
    # Save plot
    plt.savefig('model_comparison.png')
    plt.close()

if __name__ == "__main__":
    plot_comparison_bars() 