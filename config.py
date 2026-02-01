"""
Generic Configuration Module
Configuration for any review dataset with flexible column mapping
"""

import os
from typing import List, Optional, Dict


class Config:
    """
    Configuration class for generic ML pipeline
    Works with any JSON/JSONL review dataset
    """
    
    # Data paths
    DATA_PATH = None
    OUTPUT_PATH = './outputs/results.png'
    
    # Feature and label columns (user-specified)
    TEXT_COLUMN = None          # Required: column containing review text
    LABEL_COLUMN = None         # Required: column containing labels/ratings
    ADDITIONAL_FEATURES = []    # Optional: additional numeric columns
    
    # Data cleaning configuration
    ENABLE_CLEANING = True
    CLEANING_OPTIONS = {
        'missing_values': True,      # Handle missing data
        'duplicates': True,           # Remove duplicate rows
        'text_garbling': True,        # Fix text issues (spaces, special chars)
        'type_errors': True,          # Fix type inconsistencies
        'remove_empty_text': True,    # Remove rows with empty text
        'normalize_labels': False,    # Normalize label values (optional)
    }
    
    # Label processing
    LABEL_TYPE = 'auto'          # 'auto', 'multiclass', 'binary', 'regression'
    BINARY_THRESHOLD = None      # For converting to binary (e.g., >= 4 is positive)
    LABEL_MAPPING = None         # Custom label mapping dict
    
    # Data splitting parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Word2Vec parameters
    VECTOR_SIZE = 300
    WINDOW_SIZE = 5
    MIN_COUNT = 2
    WORKERS = 4
    
    # XGBoost parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = 6
    LEARNING_RATE = 0.1
    
    # Cross-validation parameters
    N_FOLDS = 5
    
    # Verbose output
    VERBOSE = True
    
    @classmethod
    def validate_config(cls):
        """Validate that required configurations are set"""
        errors = []
        
        if cls.DATA_PATH is None:
            errors.append("DATA_PATH must be specified")
        
        if cls.TEXT_COLUMN is None:
            errors.append("TEXT_COLUMN must be specified")
        
        if cls.LABEL_COLUMN is None:
            errors.append("LABEL_COLUMN must be specified")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def set_cleaning_options(cls, options: List[str]):
        """
        Set which cleaning operations to perform
        
        Args:
            options: List of cleaning operation names to enable
                    Use 'all' to enable all, 'none' to disable all
        """
        if 'none' in options:
            for key in cls.CLEANING_OPTIONS:
                cls.CLEANING_OPTIONS[key] = False
            cls.ENABLE_CLEANING = False
        elif 'all' in options:
            for key in cls.CLEANING_OPTIONS:
                cls.CLEANING_OPTIONS[key] = True
            cls.ENABLE_CLEANING = True
        else:
            # Disable all first
            for key in cls.CLEANING_OPTIONS:
                cls.CLEANING_OPTIONS[key] = False
            # Enable specified options
            for option in options:
                if option in cls.CLEANING_OPTIONS:
                    cls.CLEANING_OPTIONS[option] = True
                    cls.ENABLE_CLEANING = True
                else:
                    print(f"Warning: Unknown cleaning option '{option}' - skipping")
    
    @classmethod
    def infer_label_type(cls, unique_labels):
        """Infer label type from unique values"""
        n_unique = len(unique_labels)
        
        # Check if all labels are numeric
        try:
            numeric_labels = [float(l) for l in unique_labels]
            
            if n_unique == 2:
                return 'binary'
            elif n_unique <= 10:
                return 'multiclass'
            else:
                return 'regression'
        except (ValueError, TypeError):
            # Non-numeric labels
            if n_unique == 2:
                return 'binary'
            else:
                return 'multiclass'
    
    @classmethod
    def display_config(cls):
        """Display all configuration parameters"""
        print("\n" + "=" * 70)
        print("CONFIGURATION PARAMETERS")
        print("=" * 70)
        
        print("\nData Configuration:")
        print(f"  Data Path: {cls.DATA_PATH}")
        print(f"  Output Path: {cls.OUTPUT_PATH}")
        print(f"  Text Column: {cls.TEXT_COLUMN}")
        print(f"  Label Column: {cls.LABEL_COLUMN}")
        if cls.ADDITIONAL_FEATURES:
            print(f"  Additional Features: {cls.ADDITIONAL_FEATURES}")
        
        print("\nLabel Configuration:")
        print(f"  Label Type: {cls.LABEL_TYPE}")
        if cls.BINARY_THRESHOLD is not None:
            print(f"  Binary Threshold: {cls.BINARY_THRESHOLD}")
        if cls.LABEL_MAPPING:
            print(f"  Label Mapping: {cls.LABEL_MAPPING}")
        
        print("\nData Cleaning:")
        print(f"  Cleaning Enabled: {cls.ENABLE_CLEANING}")
        if cls.ENABLE_CLEANING:
            print("  Active Cleaning Operations:")
            for operation, enabled in cls.CLEANING_OPTIONS.items():
                if enabled:
                    print(f"    ✓ {operation}")
        
        print("\nData Splitting:")
        print(f"  Test Size: {cls.TEST_SIZE}")
        print(f"  Random State: {cls.RANDOM_STATE}")
        
        print("\nWord2Vec Parameters:")
        print(f"  Vector Size: {cls.VECTOR_SIZE}")
        print(f"  Window Size: {cls.WINDOW_SIZE}")
        print(f"  Min Count: {cls.MIN_COUNT}")
        print(f"  Workers: {cls.WORKERS}")
        
        print("\nXGBoost Parameters:")
        print(f"  N Estimators: {cls.N_ESTIMATORS}")
        print(f"  Max Depth: {cls.MAX_DEPTH}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        
        print("\nCross-Validation:")
        print(f"  N Folds: {cls.N_FOLDS}")
        print("=" * 70)


# Create default config instance
config = Config()
