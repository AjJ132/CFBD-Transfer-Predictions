import pandas as pd
from typing import List, Dict, Any
from pymongo import MongoClient

class FlattenQuarterbacks:

    def __init__(self, db_client: MongoClient, seasons: List[int]):
        self.db_client = db_client
        self.db = self.db_client['cfb_data_copy']
        self.seasons = seasons

    def gather_season_data(self, season: int) -> List[Dict[str, Any]]:
        cursor = self.db.quarterback_profiles.find({"season": season})
        return list(cursor)

    def flatten_dict(self, d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        flattened = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flattened.update(self.flatten_dict(v, f"{prefix}{k}_"))
            else:
                flattened[f"{prefix}{k}"] = v
        return flattened

    def flatten(self):
        data = []

        for season in self.seasons:
            season_data = self.gather_season_data(season)
            for item in season_data:
                flattened_item = {'season': item['season']}
                for key in ['player', 'team', 'usage', 'stats']:
                    if key in item:
                        flattened_item.update(self.flatten_dict(item[key], f"{key}_"))
                data.append(flattened_item)

        df = pd.DataFrame(data)

        #sort by season, and player_firstName/player_lastName
        df = df.sort_values(by=['player_firstName', 'player_lastName'])

        df.to_csv("data/quarterbacks.csv", index=False)

        print("Flattened quarterback data saved to data/quarterbacks.csv")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumn names:")
        print(df.columns.tolist())