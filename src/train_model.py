"""
Model training module for diabetes prediction.
Trains multiple models and saves the best one.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import download_dataset, load_and_preprocess_data

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate them.
    
    Returns:
        Dictionary of trained models with their scores
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Cross-Validation Score (mean): {cv_mean:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_score': cv_mean
        }
    
    return results

def save_best_model(results, scaler, model_dir='models'):
    """
    Save the best performing model based on ROC-AUC score.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Find best model based on ROC-AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    best_score = results[best_model_name]['roc_auc']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"ROC-AUC Score: {best_score:.4f}")
    print(f"{'='*60}")
    
    # Save model and scaler
    model_path = os.path.join(model_dir, 'diabetes_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'roc_auc': results[best_model_name]['roc_auc'],
        'cv_score': results[best_model_name]['cv_score']
    }
    
    metadata_path = os.path.join(model_dir, 'model_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metadata saved to: {metadata_path}")
    
    return best_model, best_model_name

def main():
    """
    Main training pipeline.
    """
    print("="*60)
    print("DIABETES PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Download and load data
    print("\n1. Loading data...")
    filepath = download_dataset()
    if not filepath:
        print("Failed to download dataset. Exiting.")
        return
    
    X, y, df = load_and_preprocess_data(filepath)
    
    # Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n4. Training models...")
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save best model
    print("\n5. Saving best model...")
    save_best_model(results, scaler)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
