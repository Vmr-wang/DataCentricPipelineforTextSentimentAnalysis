"""
Model Training Module
Handles XGBoost training and cross-validation
"""

import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train_xgboost_model(X_train_vectors, y_train, n_estimators=200, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost classifier
    
    Args:
        X_train_vectors: Training feature vectors
        y_train: Training labels
        n_estimators (int): Number of boosting rounds
        max_depth (int): Maximum tree depth
        learning_rate (float): Learning rate
    
    Returns:
        XGBClassifier: Trained model
    """
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING (XGBoost)")
    print("=" * 60)
    
    print("Initializing XGBoost classifier...")
    print(f"  Parameters:")
    print(f"    - n_estimators: {n_estimators}")
    print(f"    - max_depth: {max_depth}")
    print(f"    - learning_rate: {learning_rate}")
    print(f"    - objective: multi:softmax")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softmax',
        num_class=5,  # 5 rating classes (1-5)
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    print(f"\nTraining model on {len(X_train_vectors)} samples...")
    model.fit(X_train_vectors, y_train - 1)  # XGBoost expects labels starting from 0
    
    print(f"✓ Model training completed")
    
    return model


def cross_validate_model(model, X_train_vectors, y_train, n_folds=5):
    """
    Perform k-fold cross-validation
    
    Args:
        model: Trained model
        X_train_vectors: Training feature vectors
        y_train: Training labels
        n_folds (int): Number of folds
    
    Returns:
        np.array: Cross-validation scores
    """
    print("\n" + "=" * 60)
    print(f"STEP 5: {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    print(f"Performing {n_folds}-fold stratified cross-validation...")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, 
        X_train_vectors, 
        y_train - 1,  # XGBoost expects 0-based labels
        cv=skf, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"\n✓ Cross-validation completed")
    print(f"\nCross-validation results:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\nOverall CV Statistics:")
    print(f"  Mean Accuracy:  {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"  Std Deviation:  {cv_scores.std():.4f}")
    print(f"  Min Accuracy:   {cv_scores.min():.4f}")
    print(f"  Max Accuracy:   {cv_scores.max():.4f}")
    
    return cv_scores
