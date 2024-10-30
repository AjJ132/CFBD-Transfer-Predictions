import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class LongitudinalTransferPredictionModel:
    def __init__(self, x_train, y_train, x_test, y_test, player_id_col='player_playerId', season_col='season'):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.player_id_col = player_id_col
        self.season_col = season_col
        self.feature_cols = [col for col in x_train.columns if col not in [player_id_col, season_col]]
        self.model = None
        self.scaler = StandardScaler()
        self.results_dir = self.create_results_directory()

    def create_results_directory(self):
        results_dir = os.path.join('data', 'model_results', 'longitudinal_model')
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def preprocess_data(self, X, y):
        # Sort by player and season
        X = X.sort_values([self.player_id_col, self.season_col])
        
        # Create sequences for each player
        sequences = []
        labels = []
        for player_id, group in X.groupby(self.player_id_col):
            seq = group[self.feature_cols].values
            sequences.append(seq)
            labels.append(y[group.index[-1]])  # Use the last season's label
    
        # Pad sequences to have the same length
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [np.pad(seq, ((max_len - len(seq), 0), (0, 0)), mode='constant') for seq in sequences]
        
        return np.array(padded_sequences), np.array(labels)

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        # Scale features
        X_train_scaled = np.array([self.scaler.fit_transform(x) for x in X_train])
        X_val_scaled = np.array([self.scaler.transform(x) for x in X_val])

        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping]
        )
        
        return history

    def evaluate_model(self, X_test, y_test):
        X_test_scaled = np.array([self.scaler.transform(x) for x in X_test])
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
    
        # Generate predictions
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
        y_pred = y_pred.flatten()  # Flatten predictions to match y_test shape
    
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
    
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
    
        # Save metrics
        self.save_metrics(loss, accuracy, cm, report)

    def save_metrics(self, loss, accuracy, cm, report):
        # Save confusion matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()

        # Save metrics to a JSON file
        metrics = {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save a summary text file
        with open(os.path.join(self.results_dir, 'summary.txt'), 'w') as f:
            f.write(f"Test Loss: {loss:.4f}\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(self.y_test, (self.model.predict(np.array([self.scaler.transform(x) for x in self.x_test])) > 0.5).astype(int).flatten()))

    def run(self):
        X_train, y_train = self.preprocess_data(self.x_train, self.y_train)
        X_test, y_test = self.preprocess_data(self.x_test, self.y_test)
        
        print("Preprocessed Training Data Shape:", X_train.shape)
        print("Preprocessed Testing Data Shape:", X_test.shape)
        
        history = self.train_model(X_train, y_train, X_test, y_test)
        self.evaluate_model(X_test, y_test)

# Usage remains the same:
# model = LongitudinalTransferPredictionModel(x_train, y_train, x_test, y_test)
# model.run()