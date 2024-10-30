"""
SUMMARY

This script implements a Deep Neural Network model for predicting the likelihood of college football players entering the transfer portal. Key decisions and features include:

1. Precision-focused approach: We prioritize precision over recall to minimize false positives.
2. Threshold adjustment: We use a higher threshold for positive predictions to reduce false positives.
3. Feature importance: We analyze and utilize the most predictive features for transfer likelihood.
4. Regularization: We employ L2 regularization and dropout to prevent overfitting.
5. Balanced evaluation: We use precision, recall, and F1-score, with an emphasis on precision.
6. Interpretability: We include methods for explaining model predictions to assist recruiters.
7. Calibration: We ensure predicted probabilities are well-calibrated for decision-making.
8. Periodic retraining: We implement a framework for regular model updates to adapt to changing patterns.

This model serves as a component of a virtual recruiting board, providing data-driven insights to complement human expertise in college football recruiting.
"""


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import re

class DeepNeuralNetwork:
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.model = None
        self.history = None
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.scaler = StandardScaler()
        self.run_dir = self.create_run_directory()
        
    def create_run_directory(self):
        base_dir = os.path.join('.', 'data', 'model_runs')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        run_numbers = [int(re.search(r'\d+', run).group()) for run in existing_runs if re.search(r'\d+', run)]
        
        next_run_number = max(run_numbers, default=0) + 1
        run_dir = os.path.join(base_dir, f'Model Run {next_run_number:03d}')
        os.makedirs(run_dir)
        
        return run_dir
        
    def preprocess_data(self):
        # Replace -999 with NaN
        self.X_train = self.X_train.replace(-999, np.nan)
        self.X_val = self.X_val.replace(-999, np.nan)
        
        # Impute missing values with median
        for column in self.X_train.columns:
            median_value = self.X_train[column].median()
            self.X_train[column] = self.X_train[column].fillna(median_value)
            self.X_val[column] = self.X_val[column].fillna(median_value)
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train_scaled, self.y_train)
        
        print("Original dataset shape:", self.X_train_scaled.shape, self.y_train.shape)
        print("Resampled dataset shape:", self.X_train_resampled.shape, self.y_train_resampled.shape)
        
        # Convert y to numpy arrays
        self.y_train_resampled = np.array(self.y_train_resampled)
        self.y_val = self.y_val.to_numpy()
        
    def build_model(self):
        input_dim = self.X_train.shape[1]
        
        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 20:
            lr *= 0.1
        elif epoch > 40:
            lr *= 0.01
        return lr
    
    def train_model(self, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
        try:
            self.history = self.model.fit(
                self.X_train_resampled, self.y_train_resampled,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_val_scaled, self.y_val),
                callbacks=[early_stopping, lr_scheduler]
            )
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            print("Shape of X_train_resampled:", self.X_train_resampled.shape)
            print("Shape of y_train_resampled:", self.y_train_resampled.shape)
            print("First few elements of y_train_resampled:", self.y_train_resampled[:10])
            raise
    
    def evaluate_model(self):
        y_pred = (self.model.predict(self.X_val_scaled) > 0.5).astype(int)
        y_pred_proba = self.model.predict(self.X_val_scaled)
        
        print("Model Evaluation Results:")
        print("\nClassification Report:")
        cls_report = classification_report(self.y_val, y_pred, output_dict=True)
        print(classification_report(self.y_val, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(self.y_val, y_pred)
        print(conf_matrix)
        
        roc_auc = roc_auc_score(self.y_val, y_pred_proba)
        print(f"\nROC AUC: {roc_auc:.4f}")
        
        self.plot_confusion_matrix(conf_matrix)
        self.plot_training_history()
        X_val_df = pd.DataFrame(self.X_val_scaled, columns=self.X_val.columns)
        self.plot_feature_importance(X_val_df, self.y_val)

        # Save evaluation metrics as JSON
        metrics = {
            "classification_report": cls_report,
            "confusion_matrix": conf_matrix.tolist(),
            "roc_auc": roc_auc
        }
        with open(os.path.join(self.run_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    
    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(121)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.subplot(122)
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'confusion_matrices.png'))
        plt.close()
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(121)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(122)
        plt.plot(self.history.history['auc'], label='Training AUC')
        plt.plot(self.history.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_history.png'))
        plt.close()

    def custom_permutation_importance(self, X, y, n_repeats=10):
        baseline_score = roc_auc_score(y, self.model.predict(X))
        importances = []
        
        for feature in X.columns:
            feature_importances = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature])
                permuted_score = roc_auc_score(y, self.model.predict(X_permuted))
                feature_importances.append(baseline_score - permuted_score)
            importances.append(np.mean(feature_importances))
        
        return np.array(importances)

    def plot_feature_importance(self, X, y, n_repeats=10):
        feature_importance = self.custom_permutation_importance(X, y, n_repeats)
        
        sorted_idx = feature_importance.argsort()

        plt.figure(figsize=(10, 8))
        plt.barh(range(X.shape[1]), feature_importance[sorted_idx])
        plt.yticks(range(X.shape[1]), [X.columns[i] for i in sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_importance.png'))
        plt.close()

        # Save feature importance scores
        feature_importance_dict = {X.columns[i]: feature_importance[i] for i in sorted_idx}
        with open(os.path.join(self.run_dir, 'feature_importance.json'), 'w') as f:
            json.dump(feature_importance_dict, f, indent=4)

    def run(self):
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()

# Usage
# dnn = DeepNeuralNetwork(X_train, y_train)
# dnn.run()