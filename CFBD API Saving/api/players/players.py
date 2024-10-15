import pymongo
import cfbd
from cfbd.models.player_usage import PlayerUsage
from pymongo import MongoClient
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import threading
import sys

class CFBPlayerUsageExtractor:
    def __init__(self, api_client: cfbd.ApiClient, db_client: MongoClient, years: List[int], max_retries: int = 3, base_wait: float = 1.0):
        self.api_client = api_client
        self.db_client = db_client
        self.players_api = cfbd.PlayersApi(self.api_client)
        self.years = years
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.log_lock = threading.Lock()
        self.progress_bar = None

    def log_message(self, message: str):
        """Thread-safe logging function that prints below the progress bar."""
        with self.log_lock:
            if self.progress_bar:
                current_position = self.progress_bar.n
                self.progress_bar.clear()
                print(message)
                self.progress_bar.update(0)
                self.progress_bar.refresh()

    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except cfbd.ApiException as e:
                if e.status == 429 and attempt < self.max_retries - 1:
                    wait_time = self.base_wait * (2 ** attempt) + random.uniform(0, 1)
                    self.log_message(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

    def get_teams_from_db(self, year: int) -> List[Dict[str, Any]]:
        """Fetch teams from the MongoDB for a specific year."""
        db = self.db_client.get_database("cfb_data")
        collection = db["teams"]
        return list(collection.find({"season": year}))

    def get_player_usage(self, team: str, year: int) -> List[Dict[str, Any]]:
        try:
            usage_data = self.retry_with_backoff(self.players_api.get_player_usage, year=year, team=team)
            return [usage.to_dict() for usage in usage_data]
        except cfbd.ApiException as e:
            self.log_message(f"An error occurred while fetching player usage for team {team} in year {year}: {e}")
            return []

    def save_player_usage_to_db(self, usage_data: List[Dict[str, Any]], year: int, collection_name: str = "player_usage"):
        """Save the player usage data to MongoDB."""
        db = self.db_client.get_database("cfb_data")
        collection = db[collection_name]
        
        operations = [
            pymongo.UpdateOne(
                {"name": usage["name"], "team": usage["team"], "season": year},
                {"$set": {**usage, "season": year}},
                upsert=True
            ) for usage in usage_data
        ]
        result = collection.bulk_write(operations)
        # self.log_message(f"Inserted/Updated {result.upserted_count + result.modified_count} player usage records for year {year}")

    def process_team(self, team: Dict[str, Any], year: int):
        """Process a single team: fetch player usage data and save to database."""
        school = team["school"]
        usage_data = self.get_player_usage(school, year)
        if usage_data:
            self.save_player_usage_to_db(usage_data, year)
        return usage_data

    def extract_and_save_player_usage(self):
        """Main method to extract player usage data and save it to the database for all specified years."""
        for year in self.years:
            self.log_message(f"Processing player usage data for year {year}")
            teams = self.get_teams_from_db(year)
            if not teams:
                self.log_message(f"No teams were found in the database for year {year}. Skipping to next year.")
                continue

            # print(f"Got {len(teams)} teams for year {year}")
            # test_team = teams[0]
            # players_usage = self.players_api.get_player_usage_with_http_info(year=year, team=test_team['school'])
            # exit()

            #TEMP: LIMIT TO TOP 3 TEAMS
            

            with tqdm(total=len(teams), desc=f"Processing teams for {year}", file=sys.stdout) as self.progress_bar:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(self.process_team, team, year) for team in teams]
                    
                    all_usage_data = []
                    for future in as_completed(futures):
                        usage_data = future.result()
                        all_usage_data.extend(usage_data)
                        self.progress_bar.update(1)

            self.log_message(f"Processed {len(teams)} teams and {len(all_usage_data)} player usage records for year {year}.")

        self.log_message("Completed processing player usage data for all specified years.")