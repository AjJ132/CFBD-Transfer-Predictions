import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class  PrepareTestTrainSets:
    def __init__(self, data_path, train_seasons, test_seasons, target, features):
        self.data_path = data_path
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons
        self.target = target
        self.features = features

    def handle_cleaning(self, X, y):
        # Combine X and y for cleaning
        data = pd.concat([X, y], axis=1)
        
        # Drop rows with null height
        data = data.dropna(subset=['player_height'])
        
        # Impute weight using KNN
        # imputer = KNNImputer(n_neighbors=5)
        # weight_imputed = imputer.fit_transform(data[['player_weight']])
        # data.loc[:, 'player_weight'] = weight_imputed

        # Check for any remaining null values
        null_counts = data.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        
        if not columns_with_nulls.empty:
            print("Columns with null values:")
            print(columns_with_nulls)
        else:
            print("No null values remaining in the dataset.")
        
        # Split back into X and y
        X = data[self.features]
        y = data[self.target]
        
        return X, y

    def prepare_sets(self):
        # Load the data
        data = pd.read_csv(self.data_path)
        
        # Split the data into training and testing sets based on seasons
        train_data = data[data['season'].isin(self.train_seasons)]
        test_data = data[data['season'].isin(self.test_seasons)]
        
        # Filter the columns based on the features list
        X_train = train_data[self.features]
        y_train = train_data[self.target]
        X_test = test_data[self.features]
        y_test = test_data[self.target]

        print("Training data:")
        X_train, y_train = self.handle_cleaning(X_train, y_train)
        
        print("\nTest data:")
        X_test, y_test = self.handle_cleaning(X_test, y_test)

        # Save the data
        X_train.to_csv('data/sets/x_train2.csv', index=False)
        y_train.to_csv('data/sets/y_train2.csv', index=False)
        X_test.to_csv('data/sets/x_test2.csv', index=False)
        y_test.to_csv('data/sets/y_test2.csv', index=False)

        return X_train, y_train, X_test, y_test