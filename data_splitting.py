"""
Data Splitting Module
Handles train-test data splitting with support for multiple features
"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(df, test_size=0.2, random_state=42, 
               text_column='review_text', 
               label_column='rating',
               additional_features=None):
    """
    Split data into training and testing sets
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        test_size (float): Proportion of test set
        random_state (int): Random seed
        text_column (str): Name of text feature column
        label_column (str): Name of label column
        additional_features (list): List of additional feature column names
    
    Returns:
        tuple: X_text_train, X_text_test, X_add_train, X_add_test, y_train, y_test
    """
    if additional_features is None:
        additional_features = []
    
    print("\n" + "=" * 60)
    print("STEP 2: DATA SPLITTING")
    print("=" * 60)

    # Extract text features
    X_text = df[text_column]
    
    # Extract additional numeric features if specified
    X_additional = None
    if additional_features:
        available_features = [f for f in additional_features if f in df.columns]
        if available_features:
            X_additional = df[available_features]
            print(f"Using additional features: {available_features}")
        else:
            print(f"Warning: Additional features {additional_features} not found in dataframe")
    
    # Extract labels
    y = df[label_column]
    
    # Create indices for splitting
    indices = df.index
    
    # Try stratified split first, fall back to regular split if it fails
    try:
        # Stratified split to maintain class distribution
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
    except (TypeError, ValueError) as e:
        # If stratification fails (e.g., due to unhashable types or too few samples per class)
        print(f"Warning: Stratified split failed ({e}), using regular split")
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size, 
            random_state=random_state
        )
    
    # Split text features
    X_text_train = X_text.loc[idx_train]
    X_text_test = X_text.loc[idx_test]
    
    # Split additional features
    X_add_train = X_additional.loc[idx_train] if X_additional is not None else None
    X_add_test = X_additional.loc[idx_test] if X_additional is not None else None
    
    # Split labels
    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]
    
    print(f"✓ Split data into {(1-test_size)*100:.0f}% training and {test_size*100:.0f}% testing")
    print(f"  Training set size: {len(X_text_train)}")
    print(f"  Testing set size:  {len(X_text_test)}")
    if X_additional is not None:
        print(f"  Additional features: {list(X_additional.columns)}")
    
    # Display distribution in training set
    print(f"\nTraining set label distribution:")
    train_dist = y_train.value_counts().sort_index()
    for rating, count in train_dist.items():
        percentage = (count / len(y_train)) * 100
        print(f"  {rating}: {count:5d} ({percentage:5.2f}%)")
    
    # Display distribution in testing set
    print(f"\nTesting set label distribution:")
    test_dist = y_test.value_counts().sort_index()
    for rating, count in test_dist.items():
        percentage = (count / len(y_test)) * 100
        print(f"  {rating}: {count:5d} ({percentage:5.2f}%)")
    
    return X_text_train, X_text_test, X_add_train, X_add_test, y_train, y_test
