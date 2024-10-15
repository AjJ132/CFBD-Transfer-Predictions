import pymongo
import cfbd
from cfbd.rest import ApiException
from pymongo import MongoClient
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import threading
import sys

class CFBPlayerStatsExtractor:
    def __init__(self, api_client: cfbd.ApiClient, db_client: MongoClient, years: List[int], max_retries: int = 3, base_wait: float = 1.0):
        self.api_client = api_client
        self.db_client = db_client
        self.stats_api = cfbd.StatsApi(self.api_client)
        self.teams_api = cfbd.TeamsApi(self.api_client)
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
            except ApiException as e:
                if e.status == 429 and attempt < self.max_retries - 1:
                    wait_time = self.base_wait * (2 ** attempt) + random.uniform(0, 1)
                    self.log_message(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

    def get_teams_from_db(self, year: int) -> List[Dict[str, Any]]:
        """Fetch teams from the database for a specific year."""
        db = self.db_client.get_database("cfb_data")
        teams_collection = db["teams"]
        return list(teams_collection.find({"season": year}))

    def get_player_stats(self, year: int, team: str) -> List[Dict[str, Any]]:
        """Fetch player stats for a specific year and team."""
        try:
            stats = self.retry_with_backoff(self.stats_api.get_player_season_stats, year=year, team=team)
            return [stat.to_dict() for stat in stats]
        except ApiException as e:
            self.log_message(f"An error occurred while fetching player stats for {team} in year {year}: {e}")
            return []

    def save_player_stats_to_db(self, stats: List[Dict[str, Any]], year: int, collection_name: str = "player_stats"):
        """Save the player stats to MongoDB."""
        db = self.db_client.get_database("cfb_data")
        collection = db[collection_name]
        
        operations = [
            pymongo.UpdateOne(
                {"playerId": stat["playerId"], "team": stat["team"], "season": stat["season"], 'category': stat['category'], 'statType': stat['statType']},
                {"$set": stat},
                upsert=True
            ) for stat in stats
        ]
        result = collection.bulk_write(operations)
        self.log_message(f"Inserted/Updated {result.upserted_count + result.modified_count} player stats for year {year}")

    def process_team_stats(self, team: Dict[str, Any], year: int):
        """Process stats for a single team: fetch stats and save to database."""
        school = team["school"]
        stats = self.get_player_stats(year, school)
        if stats:
            self.save_player_stats_to_db(stats, year)
        return stats

    def extract_and_save_player_stats(self):
        """Main method to extract player stats and save them to the database for all specified years."""
        for year in self.years:
            self.log_message(f"Processing player stats for year {year}")
            teams = self.get_teams_from_db(year)
            if not teams:
                self.log_message(f"No teams were found in the database for year {year}. Skipping to next year.")
                continue

            # teams = [team for team in teams if team["school"] in ["Georgia", "Alabama"]]

            with tqdm(total=len(teams), desc=f"Processing team stats for {year}", file=sys.stdout) as self.progress_bar:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(self.process_team_stats, team, year) for team in teams]
                    
                    all_stats = []
                    for future in as_completed(futures):
                        team_stats = future.result()
                        all_stats.extend(team_stats)
                        self.progress_bar.update(1)

            self.log_message(f"Processed {len(teams)} teams and {len(all_stats)} player stats for year {year}.")

        self.log_message("Completed processing player stats for all specified years.")

