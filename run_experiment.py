"""
Generic Experiment Runner
Works with any JSON/JSONL review dataset with support for multiple features
"""

from data_cleaning import load_and_clean_data, load_data_without_cleaning
from data_splitting import split_data
from feature_extraction import train_word2vec, transform_texts_to_vectors
from model_training import train_xgboost_model, cross_validate_model
from model_testing import evaluate_model
from visualization import plot_results
import numpy as np


def run_experiment(data_path,
                   output_path,
                   text_column,
                   label_column,
                   additional_features=None,
                   enable_cleaning=True,
                   cleaning_options=None,
                   test_size=0.2,
                   vector_size=300,
                   window=5,
                   min_count=2,
                   n_estimators=200,
                   max_depth=6,
                   learning_rate=0.1,
                   n_folds=5,
                   verbose=True):
    """
    Run the complete Word2Vec + XGBoost experiment on any review dataset
    
    Args:
        data_path (str): Path to data file (JSON or JSONL)
        output_path (str): Path to save results
        text_column (str): Name of text feature column
        label_column (str): Name of label column
        additional_features (list): List of additional feature column names
        enable_cleaning (bool): Whether to perform data cleaning
        cleaning_options (dict): Dict of cleaning operations to perform
        test_size (float): Test set proportion
        vector_size (int): Word2Vec vector dimension
        window (int): Word2Vec context window
        min_count (int): Word2Vec minimum word frequency
        n_estimators (int): XGBoost number of trees
        max_depth (int): XGBoost max tree depth
        learning_rate (float): XGBoost learning rate
        n_folds (int): Number of CV folds
        verbose (bool): Whether to print detailed output
    
    Returns:
        dict: Complete experiment results
    """
    if additional_features is None:
        additional_features = []
    
    if verbose:
        print("\n" + "#" * 60)
        print("# GENERIC REVIEW ANALYSIS PIPELINE")
        print("#" * 60)
        print(f"# Text Column: {text_column}")
        print(f"# Label Column: {label_column}")
        if additional_features:
            print(f"# Additional Features: {additional_features}")
        print(f"# Cleaning: {'ENABLED' if enable_cleaning else 'DISABLED'}")
        print("#" * 60)
    
    # Step 1: Load and optionally clean data
    if enable_cleaning and cleaning_options:
        df = load_and_clean_data(
            data_path=data_path,
            text_column=text_column,
            label_column=label_column,
            additional_features=additional_features,
            cleaning_options=cleaning_options,
            verbose=verbose
        )
    else:
        df = load_data_without_cleaning(
            data_path=data_path,
            text_column=text_column,
            label_column=label_column,
            verbose=verbose
        )
    
    # Show dataset statistics
    if verbose:
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Text column: {text_column}")
        print(f"Label column: {label_column}")
        if additional_features:
            available_add = [f for f in additional_features if f in df.columns]
            print(f"Additional features: {available_add}")
        print(f"\nLabel distribution:")
        label_dist = df[label_column].value_counts().sort_index()
        for label, count in label_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count:5d} ({percentage:5.2f}%)")
    
    # Step 2: Split data
    X_text_train, X_text_test, X_add_train, X_add_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        text_column=text_column,
        label_column=label_column,
        additional_features=additional_features
    )
    
    # Step 3: Train Word2Vec and extract features
    w2v_model = train_word2vec(
        X_text_train,
        vector_size=vector_size,
        window=window,
        min_count=min_count
    )
    
    # Transform texts to vectors
    X_text_train_vectors = transform_texts_to_vectors(X_text_train, w2v_model, vector_size)
    X_text_test_vectors = transform_texts_to_vectors(X_text_test, w2v_model, vector_size)
    
    # Step 4: Combine text vectors with additional features
    if X_add_train is not None:
        if verbose:
            print(f"\nCombining text vectors with {X_add_train.shape[1]} additional features...")
        X_train_vectors = np.hstack([X_text_train_vectors, X_add_train.values])
        X_test_vectors = np.hstack([X_text_test_vectors, X_add_test.values])
        if verbose:
            print(f"✓ Combined feature shape: {X_train_vectors.shape}")
    else:
        X_train_vectors = X_text_train_vectors
        X_test_vectors = X_text_test_vectors
    
    # Step 5: Train XGBoost model
    model = train_xgboost_model(
        X_train_vectors, y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    
    # Step 6: Cross-validation
    cv_scores = cross_validate_model(model, X_train_vectors, y_train, n_folds=n_folds)
    
    # Step 7: Evaluate on test set
    results = evaluate_model(model, X_test_vectors, y_test)
    
    # Step 8: Visualize results
    plot_results(df, cv_scores, results, output_path)
    
    if verbose:
        print("\n" + "#" * 60)
        print("# EXPERIMENT COMPLETED SUCCESSFULLY")
        print("#" * 60)
    
    # Return all components
    return {
        'dataframe': df,
        'word2vec_model': w2v_model,
        'xgboost_model': model,
        'cv_scores': cv_scores,
        'test_results': results,
        'train_vectors': X_train_vectors,
        'test_vectors': X_test_vectors
    }
