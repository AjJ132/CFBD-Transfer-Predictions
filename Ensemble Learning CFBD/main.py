import pandas as pd
from scripts.prepare_test_train_sets import PrepareTestTrainSets
from scripts.ensemble_learn import EnsembleLearning
from scripts.deep_nueral_network import DeepNeuralNetwork
from scripts.enhanced_nueral_network import EnhancedDeepNeuralNetwork
import numpy as np

def main():

    train_seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    test_seasons = [2023]

    data_path = 'data/raw/qbs_ml_ready.csv'

    features = [
        "player_height",
        "player_weight",
        "player_class",
        "usage_overall",
        "stats_passing_pct",
        "win_rate",  
        "is_power_5",
        "class_risk_factor",
        "win_rate_class_risk_factor",
        "is_not_same_coach_next_season",
        "yoy_win_rate",
        "yoy_stats_passing_pct",
        "yoy_usage_overall",
        "team_coach_encoded",
        "team_name_encoded",
        "team_conference_ACC",
        "team_conference_American Athletic",
        "team_conference_Big 12",
        "team_conference_Big Ten",
        "team_conference_Conference USA",
        "team_conference_FBS Independents",
        "team_conference_Mid-American",
        "team_conference_Mountain West",
        "team_conference_Pac-12",
        "team_conference_SEC",
        "team_conference_Sun Belt"
    ]

    prepare_sets = PrepareTestTrainSets(data_path, train_seasons, test_seasons, 'transfer', features)
    x_train, y_train, x_test, y_test = prepare_sets.prepare_sets()
    # x_test = pd.read_csv('data/sets/x_test.csv')
    # y_test = pd.read_csv('data/sets/y_test.csv')
    # x_train = pd.read_csv('data/sets/x_train.csv')
    # y_train = pd.read_csv('data/sets/y_train.csv')
    

    # Print data info (optional, for debugging)
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("Columns: " + ", ".join(x_train.columns))
    print("Class distribution in y_train:")
    print(y_train.value_counts(normalize=True))
    

    # Initialize and run the Deep Neural Network
    # dnn = DeepNeuralNetwork(x_train, y_train)
    # dnn.run()

    enhanced_dnn = EnhancedDeepNeuralNetwork(x_train, y_train, x_test, y_test, target_precision=0.5, max_iterations=200)
    enhanced_dnn.run()



if __name__ == '__main__':
    main()