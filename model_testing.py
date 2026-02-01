"""
Model Testing Module
Handles model evaluation and testing
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def evaluate_model(model, X_test_vectors, y_test):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        X_test_vectors: Test feature vectors
        y_test: True test labels
    
    Returns:
        dict: Evaluation results
    """
    print("\n" + "=" * 60)
    print("STEP 6: MODEL TESTING AND EVALUATION")
    print("=" * 60)
    
    print(f"Making predictions on {len(X_test_vectors)} test samples...")
    
    # Make predictions (add 1 back to convert from 0-based to 1-based)
    y_pred = model.predict(X_test_vectors) + 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Predictions completed")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted F1-Score:  {f1:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class metrics
    print(f"Per-Class Accuracy:")
    classes = sorted(y_test.unique())
    for cls in classes:
        mask = y_test == cls
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  Rating {cls}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_test': y_test,
        'confusion_matrix': cm
    }
    
    return results
