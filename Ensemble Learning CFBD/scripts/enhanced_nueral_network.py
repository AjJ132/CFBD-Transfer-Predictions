import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, accuracy_score
from imblearn.over_sampling import SMOTE
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import re

class EnhancedDeepNeuralNetwork:
    def __init__(self, X_train, y_train, X_test, y_test, target_precision=0.5, max_iterations=50):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.target_precision = target_precision
        self.max_iterations = max_iterations
        self.best_model = None
        self.best_features = None
        self.best_precision = 0
        self.feature_sets = []
        self.results = []

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

    def preprocess_data(self, X):
        X = X.replace(-999, np.nan)
        for column in X.columns:
            X[column] = X[column].fillna(X[column].median())
        return X

    def build_model(self, input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model

    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 20:
            lr *= 0.1
        elif epoch > 40:
            lr *= 0.01
        return lr

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
        model = self.build_model(X_train.shape[1])
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        history = model.fit(
            X_train_resampled, y_train_resampled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, lr_scheduler]
        )
        
        return model, history

    def evaluate_model(self, model, X_val, y_val):
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        precision = precision_score(y_val, y_pred)
        auc = roc_auc_score(y_val, model.predict(X_val))
        return precision, auc

    def run_feature_selection(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.X_train.columns)
        X_test_scaled = scaler.transform(self.X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.X_test.columns)

        for i in range(self.max_iterations):
            if i == 0:
                feature_set = list(self.X_train.columns)
            else:
                # Randomly select a subset of features
                n_features = np.random.randint(3, len(self.X_train.columns))
                feature_set = np.random.choice(self.X_train.columns, n_features, replace=False)

            X_train_subset = X_train_scaled[feature_set]
            X_val_subset = X_test_scaled[feature_set]
            
            model, history = self.train_model(X_train_subset, self.y_train, X_val_subset, self.y_test)
            precision, auc = self.evaluate_model(model, X_val_subset, self.y_test)

            self.feature_sets.append(feature_set)
            self.results.append({'precision': precision, 'auc': auc, 'n_features': len(feature_set)})

            if precision > self.best_precision:
                self.best_precision = precision
                self.best_model = model
                self.best_features = feature_set

            print(f"Iteration {i+1}: Precision = {precision:.4f}, AUC = {auc:.4f}, Features = {len(feature_set)}")

            if precision >= self.target_precision:
                print(f"Target precision reached after {i+1} iterations.")
                break

        self.save_results()

    def save_results(self):
        run_dir = self.create_run_directory()
        
        # Save best model
        self.best_model.save(os.path.join(run_dir, 'best_model.h5'))
        
        # Save best features
        with open(os.path.join(run_dir, 'best_features.json'), 'w') as f:
            json.dump(list(self.best_features), f)
        
        # Save all results
        results_df = pd.DataFrame(self.results)
        results_df['features'] = self.feature_sets
        results_df.to_csv(os.path.join(run_dir, 'all_results.csv'), index=False)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.results) + 1), [r['precision'] for r in self.results], label='Precision')
        plt.plot(range(1, len(self.results) + 1), [r['auc'] for r in self.results], label='AUC')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Model Performance Over Iterations')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'performance_over_iterations.png'))
        plt.close()

    def run(self):
        self.run_feature_selection()
        print(f"Best precision achieved: {self.best_precision:.4f}")
        print(f"Best features: {self.best_features}")
        
        # Evaluate on test set
        X_test_best = self.X_test[self.best_features]
        scaler = StandardScaler()
        X_test_best_scaled = scaler.fit_transform(X_test_best)
        
        y_pred = (self.best_model.predict(X_test_best_scaled) > 0.5).astype(int)
        
        # Calculate and print accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Generate and plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.create_run_directory(), 'confusion_matrix.png'))
        plt.close()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
