import pymongo
import cfbd 
from cfbd.models.team import Team 
from cfbd.models.roster_player import RosterPlayer
from pymongo import MongoClient
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import threading
import sys

class CFBTeamExtractor:
    def __init__(self, api_client: cfbd.ApiClient, db_client: MongoClient, years: List[int], max_retries: int = 3, base_wait: float = 1.0):
        self.api_client = api_client
        self.db_client = db_client
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
            except cfbd.ApiException as e:
                if e.status == 429 and attempt < self.max_retries - 1:
                    wait_time = self.base_wait * (2 ** attempt) + random.uniform(0, 1)
                    self.log_message(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

    def get_teams(self, year: int) -> List[Dict[str, Any]]:
        """Fetch teams from the CFB API for a specific year and return as a list of dictionaries."""
        try:
            teams = self.retry_with_backoff(self.teams_api.get_fbs_teams, year=year)
            return [team.to_dict() for team in teams]
        except cfbd.ApiException as e:
            self.log_message(f"An error occurred while fetching teams for year {year}: {e}")
            return []
        
    def save_teams_to_db(self, teams: List[Dict[str, Any]], season: int, collection_name: str = "teams"):
        """Save the team data to MongoDB."""
        db = self.db_client.get_database("cfb_data")
        collection = db[collection_name]
        
        operations = [
            pymongo.UpdateOne(
                {"id": team["id"], "season": season},
                {"$set": {**team, "season": season}},
                upsert=True
            ) for team in teams
        ]
        result = collection.bulk_write(operations)
        self.log_message(f"Inserted/Updated {result.upserted_count + result.modified_count} teams for year {season}")

    def get_team_roster(self, school: str, season: int, team_id: int) -> List[Dict[str, Any]]:
        """Fetch the roster for a specific team and year with retry logic, and add team_id to each player."""
        for attempt in range(self.max_retries):
            try:
                roster = self.teams_api.get_roster(team=school, year=season)
                return [{**player.to_dict(), "team_id": team_id} for player in roster]
            except cfbd.ApiException as e:
                if e.status == 429 and attempt < self.max_retries - 1:
                    wait_time = self.base_wait * (2 ** attempt) + random.uniform(0, 1)
                    self.log_message(f"Rate limit hit for {school} in {season}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    self.log_message(f"An error occurred while fetching roster for {school} in season {season}: {e}")
                    return []
        self.log_message(f"Failed to fetch roster for {school} in season {season} after {self.max_retries} attempts.")
        return []

    def save_players_to_db(self, players: List[Dict[str, Any]], season: int, collection_name: str = "players"):
        """Save the player data to MongoDB."""
        db = self.db_client.get_database("cfb_data")
        collection = db[collection_name]
        
        operations = [
            pymongo.UpdateOne(
                {"id": player["id"], "season": season},
                {"$set": {**player, "season": season}},
                upsert=True
            ) for player in players
        ]
        result = collection.bulk_write(operations)
        # self.log_message(f"Inserted/Updated {result.upserted_count + result.modified_count} players for season {season}")

    def process_team(self, team: Dict[str, Any], year: int):
        """Process a single team: fetch roster and save to database."""
        school = team["school"]
        team_id = team["id"]
        roster = self.get_team_roster(school, year, team_id)

        #for each player in the roster, if firstName or lastName is null or = "" then remove
        for player in roster:
            if player["firstName"] == "" or player["firstName"] == None:
                player["firstName"] = "Unknown"
            if player["lastName"] == "" or player["lastName"] == None:
                player["lastName"] = "Unknown"

        #foreach player, if year is greater than 10, then remove that player
        """
        This fixes a discrepancy in the data where some players have a year value of the current season. 
        Also the rest of their data is null
        """
        for player in roster:
            if player["year"] >= 10:
                roster.remove(player)

        if roster:
            self.save_players_to_db(roster, year)
        return roster

    def extract_and_save_teams(self):
        """Main method to extract team data and save it to the database for all specified years."""
        for year in self.years:
            self.log_message(f"Processing data for year {year}")
            teams = self.get_teams(year)
            if not teams:
                self.log_message(f"No teams were fetched for year {year}. Skipping to next year.")
                continue

            self.save_teams_to_db(teams, year)

            with tqdm(total=len(teams), desc=f"Processing teams for {year}", file=sys.stdout) as self.progress_bar:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(self.process_team, team, year) for team in teams]
                    
                    all_players = []
                    for future in as_completed(futures):
                        roster = future.result()
                        all_players.extend(roster)
                        self.progress_bar.update(1)

            self.log_message(f"Processed {len(teams)} teams and {len(all_players)} players for year {year}.")

        self.log_message("Completed processing for all specified years.")