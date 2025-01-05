import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import os
import logging
from data_processing import load_and_preprocess_data, create_data_loaders
from model import TGANet, BaselineBERTClassifier
from train_eval import Trainer, evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')

        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Create save directory if it doesn't exist
        os.makedirs(args.save_dir, exist_ok=True)

        # Load tokenizer
        logger.info('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load and preprocess data
        logger.info('Loading and preprocessing data...')
        train_data, val_data, test_data = load_and_preprocess_data(
            args.train_path,
            args.test_path,
            val_size=args.val_size,
            random_state=args.seed
        )

        # Create data loaders
        logger.info('Creating data loaders...')
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data,
            val_data,
            test_data,
            tokenizer,
            batch_size=args.batch_size
        )

        # Initialize model
        logger.info(f'Initializing {args.model_type} model...')
        if args.model_type == 'tganet':
            model = TGANet(num_labels=3)
        else:
            model = BaselineBERTClassifier(num_labels=3)
        model = model.to(device)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        total_steps = len(train_loader) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            scheduler=scheduler,
            save_dir=args.save_dir
        )

        # Train model
        logger.info('Starting training...')
        best_model_path = trainer.train()

        # Evaluate model
        logger.info('\nEvaluating best model...')
        test_report, test_conf_matrix, test_pr_curves = evaluate_model(trainer, best_model_path)

        logger.info('\nTraining completed!')
        logger.info(f'Model checkpoints and evaluation results saved in: {args.save_dir}')

    except Exception as e:
        logger.error(f'An error occurred: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stance Detection Training')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='tganet', choices=['tganet', 'baseline'],
                        help='Model type to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    
    # Optimizer arguments
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints and results')
    
    args = parser.parse_args()
    
    main(args)
