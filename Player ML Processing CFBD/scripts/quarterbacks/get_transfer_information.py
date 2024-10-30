import pandas as pd
from typing import List, Dict, Any
from pymongo import MongoClient

class QuarterbackTransferInformation:

    def __init__(self, db_client: MongoClient, seasons: List[int]):
        self.db_client = db_client
        self.db = self.db_client['cfb_data_copy']
        self.seasons = seasons

    def get_transfer_information(self, ):
        # Load data
        data = pd.read_csv("data/quarterbacks.csv")

        #add column called 'transfer' to indicate if player transferred set all to zero
        data['transfer'] = 0

        #foreach row, group by player_firstName and player_lastName
        #then foreach season see if the next season's player_teamName is different than the current season's player_teamName
        #if it is, then set the transfer column to 1, else zero. if there is no next season, set to zero
        for index, row in data.iterrows():
            player_firstName = row['player_firstName']
            player_lastName = row['player_lastName']
            player_teamName = row['player_teamName']
            season = row['season']

            #get the next season
            next_season = season + 1

            #check if the next season is in the list of seasons
            if next_season in self.seasons:
                next_season_data = data[(data['player_firstName'] == player_firstName) & (data['player_lastName'] == player_lastName) & (data['season'] == next_season)]
                if not next_season_data.empty:
                    next_season_teamName = next_season_data['player_teamName'].values[0]
                    if player_teamName != next_season_teamName:
                        data.at[index, 'transfer'] = 1

        #move transfer column to the front
        transfer_column = data.pop('transfer')

        data.insert(0, 'transfer', transfer_column)

        #print the number of transfers and non-transfers
        print("Number of transfers: ", len(data[data['transfer'] == 1]))
        print("Number of non-transfers: ", len(data[data['transfer'] == 0]))

        #print average number for usage_pass and stats_passing_pct for transfers and non-transfers
        print("Average usage_pass for transfers: ", data[data['transfer'] == 1]['usage_overall'].mean())
        print("Average usage_pass for non-transfers: ", data[data['transfer'] == 0]['usage_overall'].mean())

        print("Average stats_passing_pct for transfers: ", data[data['transfer'] == 1]['stats_passing_pct'].mean())
        print("Average stats_passing_pct for non-transfers: ", data[data['transfer'] == 0]['stats_passing_pct'].mean())

        #get the top three most transferred from teams from player_teamName
        print("Top three most transferred from teams: ", data[data['transfer'] == 1]['player_teamName'].value_counts().head(5))

        #save the data
        data.to_csv("data/prepped_quarterbacks.csv", index=False)

    