import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class EnsembleLearning:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.y_pred = None
        self.y_pred_proba = None

    def preprocess_data(self, balance_classes=True):
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features)
            ])

        if balance_classes:
            self.model = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        else:
            self.model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])

    def train_model(self, balance_classes=True):
        self.preprocess_data(balance_classes)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)

        print("Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

        print("\nConfusion Matrix:")
        print(conf_matrix)

        self.plot_confusion_matrix(conf_matrix)

        self.plot_precision_recall_curve(self.y_test, self.y_pred_proba)
        self.plot_learning_curves(self.model, self.X_train, self.y_train)
        self.plot_feature_importance(self.model, self.X_train.columns)


    # / ---------------------PLOTTING FUNCTIONS--------------------- /
    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(12, 5))
        
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
        plt.savefig('data/plots/confusion_matrices.png')
        plt.close()

    def plot_feature_importance(self, model, feature_names):
        # Get feature importances
        importances = model.named_steps['classifier'].feature_importances_
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        feature_names_preprocessed = []
        
        for name, _, column_names in preprocessor.transformers_:
            if name == 'num':
                feature_names_preprocessed.extend(column_names)
            elif name == 'cat':
                # If there are categorical features, get their one-hot encoded names
                onehot_encoder = preprocessor.named_transformers_['cat']
                categorical_feature_names = onehot_encoder.get_feature_names_out(column_names).tolist()
                feature_names_preprocessed.extend(categorical_feature_names)
        
        # Ensure the number of feature names matches the number of importances
        if len(feature_names_preprocessed) != len(importances):
            raise ValueError(f"Number of feature names ({len(feature_names_preprocessed)}) "
                            f"doesn't match number of importance values ({len(importances)})")
        
        # Sort features by importance
        feature_importance = pd.DataFrame({
            'feature': feature_names_preprocessed,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(30), x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('./data/plots/feature_importances.png')
        plt.close()

    def plot_learning_curves(self, model, X, y):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('data/plots/learning_curves.png')
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        
        plt.figure(figsize=(10, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
        plt.tight_layout()
        plt.savefig('data/plots/precision_recall_curve.png')
        plt.close()

    # / ---------------------PLOTTING FUNCTIONS--------------------- /

    def run(self, balance_classes=True):
        self.train_model(balance_classes)
        self.evaluate_model()