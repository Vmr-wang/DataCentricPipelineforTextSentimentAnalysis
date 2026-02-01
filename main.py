"""
Generic Review Analysis Pipeline - Main Entry Point
Works with any JSON/JSONL review dataset
"""

import argparse
import warnings
import os
import sys
from config import Config
from run_experiment import run_experiment

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generic ML Pipeline for Review Sentiment Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required arguments
  python main.py \\
      --data-path reviews.json \\
      --text-column review_text \\
      --label-column rating
  
  # With cleaning options
  python main.py \\
      --data-path reviews.jsonl \\
      --text-column text \\
      --label-column score \\
      --cleaning all
  
  # With custom hyperparameters
  python main.py \\
      --data-path reviews.json \\
      --text-column content \\
      --label-column sentiment \\
      --vector-size 200 \\
      --n-estimators 300
  
  # Without cleaning (for pre-cleaned data)
  python main.py \\
      --data-path clean_reviews.json \\
      --text-column review \\
      --label-column rating \\
      --cleaning none
  
  # With additional features
  python main.py \\
      --data-path reviews.json \\
      --text-column text \\
      --label-column rating \\
      --additional-features helpful_votes verified_purchase
        """
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to the input data file (JSON or JSONL)'
    )
    
    required.add_argument(
        '--text-column',
        type=str,
        required=True,
        help='Name of the column containing review text'
    )
    
    required.add_argument(
        '--label-column',
        type=str,
        required=True,
        help='Name of the column containing labels/ratings'
    )
    
    # Optional data configuration
    parser.add_argument(
        '--output-path',
        type=str,
        default='./outputs/results.png',
        help='Path to save output visualizations (default: ./outputs/results.png)'
    )
    
    parser.add_argument(
        '--additional-features',
        type=str,
        nargs='*',
        default=[],
        help='Names of additional numeric feature columns to include'
    )
    
    # Data cleaning options
    parser.add_argument(
        '--cleaning',
        type=str,
        nargs='+',
        default=['all'],
        choices=[
            'all', 'none',
            'missing_values', 'duplicates', 'text_garbling',
            'type_errors', 'remove_empty_text', 'normalize_labels'
        ],
        help='Data cleaning operations to perform (default: all)'
    )
    
    # Data splitting
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Word2Vec parameters
    parser.add_argument(
        '--vector-size',
        type=int,
        default=300,
        help='Word2Vec vector dimension (default: 300)'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='Word2Vec context window size (default: 5)'
    )
    
    parser.add_argument(
        '--min-count',
        type=int,
        default=2,
        help='Word2Vec minimum word frequency (default: 2)'
    )
    
    # XGBoost parameters
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of XGBoost trees (default: 200)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum depth of XGBoost trees (default: 6)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='XGBoost learning rate (default: 0.1)'
    )
    
    # Cross-validation
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    # Output control
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    return parser.parse_args()


def main():
    """Main function to execute the ML pipeline"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure based on arguments
    Config.DATA_PATH = args.data_path
    Config.OUTPUT_PATH = args.output_path
    Config.TEXT_COLUMN = args.text_column
    Config.LABEL_COLUMN = args.label_column
    Config.ADDITIONAL_FEATURES = args.additional_features if args.additional_features else []
    
    # Validate configuration
    try:
        Config.validate_config()
    except ValueError as e:
        print(f"\n❌ Configuration Error:\n{e}\n")
        sys.exit(1)
    
    # Configure cleaning options
    Config.set_cleaning_options(args.cleaning)
    
    # Set other parameters
    Config.TEST_SIZE = args.test_size
    Config.RANDOM_STATE = args.random_state
    Config.VECTOR_SIZE = args.vector_size
    Config.WINDOW_SIZE = args.window
    Config.MIN_COUNT = args.min_count
    Config.N_ESTIMATORS = args.n_estimators
    Config.MAX_DEPTH = args.max_depth
    Config.LEARNING_RATE = args.learning_rate
    Config.N_FOLDS = args.n_folds
    Config.VERBOSE = not args.quiet
    
    # Validate data file exists
    if not os.path.exists(Config.DATA_PATH):
        print(f"\n❌ Error: Data file not found: {Config.DATA_PATH}\n")
        sys.exit(1)
    
    # Display configuration
    if Config.VERBOSE:
        Config.display_config()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(Config.OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment
    try:
        experiment_results = run_experiment(
            data_path=Config.DATA_PATH,
            output_path=Config.OUTPUT_PATH,
            text_column=Config.TEXT_COLUMN,
            label_column=Config.LABEL_COLUMN,
            additional_features=Config.ADDITIONAL_FEATURES,
            enable_cleaning=Config.ENABLE_CLEANING,
            cleaning_options=Config.CLEANING_OPTIONS,
            test_size=Config.TEST_SIZE,
            vector_size=Config.VECTOR_SIZE,
            window=Config.WINDOW_SIZE,
            min_count=Config.MIN_COUNT,
            n_estimators=Config.N_ESTIMATORS,
            max_depth=Config.MAX_DEPTH,
            learning_rate=Config.LEARNING_RATE,
            n_folds=Config.N_FOLDS,
            verbose=Config.VERBOSE
        )
    except Exception as e:
        print(f"\n❌ Error during experiment execution:\n{e}\n")
        import traceback
        if Config.VERBOSE:
            traceback.print_exc()
        sys.exit(1)
    
    # Print final summary
    if Config.VERBOSE:
        print("\n" + "=" * 70)
        print("FINAL EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"Data File: {Config.DATA_PATH}")
        print(f"Text Column: {Config.TEXT_COLUMN}")
        print(f"Label Column: {Config.LABEL_COLUMN}")
        print(f"Model: Word2Vec + XGBoost")
        print(f"Word2Vec Vector Size: {Config.VECTOR_SIZE}")
        print(f"Word2Vec Vocabulary: {len(experiment_results['word2vec_model'].wv)} words")
        print(f"XGBoost Trees: {Config.N_ESTIMATORS}")
        print(f"\nPerformance Metrics:")
        print(f"  ✓ CV Mean Accuracy:  {experiment_results['cv_scores'].mean():.4f} ({experiment_results['cv_scores'].mean()*100:.2f}%)")
        print(f"  ✓ Test Accuracy:     {experiment_results['test_results']['accuracy']:.4f} ({experiment_results['test_results']['accuracy']*100:.2f}%)")
        print(f"  ✓ Test F1-Score:     {experiment_results['test_results']['f1_score']:.4f}")
        print(f"\n✓ All results and visualizations saved to: {Config.OUTPUT_PATH}")
        print("=" * 70)
    
    return experiment_results


if __name__ == "__main__":
    results = main()
