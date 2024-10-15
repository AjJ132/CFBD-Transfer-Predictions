import pymongo
import cfbd
from pymongo import MongoClient
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import threading
import sys

class CFBCoachesExtractor:
    def __init__(self, api_client: cfbd.ApiClient, db_client: MongoClient, years: List[int], max_retries: int = 3, base_wait: float = 1.0):
        self.api_client = api_client
        self.db_client = db_client
        self.coaches_api = cfbd.CoachesApi(self.api_client)
        self.years = years
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.log_lock = threading.Lock()
        self.progress_bar = None


    def log_message(self, message: str):
        """Thread-safe logging function that prints below the progress bar."""
        with self.log_lock:
            if self.progress_bar:
                self.progress_bar.clear()
                print(message)
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

    def get_coaches(self, year: int) -> List[Dict[str, Any]]:
        """Fetch coaches data for a specific year."""
        try:
            coaches_data = self.retry_with_backoff(self.coaches_api.get_coaches, year=year)
            return [coach.to_dict() for coach in coaches_data]
        except cfbd.ApiException as e:
            self.log_message(f"An error occurred while fetching coaches for year {year}: {e}")
            return []

    def save_coaches_to_db(self, coaches_data: List[Dict[str, Any]], year: int, collection_name: str = "coaches"):
        """Save the coaches data to MongoDB."""
        db = self.db_client.get_database("cfb_data")
        collection = db[collection_name]
        
        operations = []
        for coach in coaches_data:
            coach_doc = {
                "firstName": coach["firstName"],
                "lastName": coach["lastName"],
                "hireDate": coach.get("hireDate"),
                "seasons": []
            }
            
            for season in coach.get("seasons", []):
                if season["year"] == year:
                    coach_doc["seasons"].append({
                        "school": season["school"],
                        "year": season["year"],
                        "games": season["games"],
                        "wins": season["wins"],
                        "losses": season["losses"],
                        "ties": season["ties"],
                        "preseasonRank": season.get("preseason_rank"),
                        "postseasonRank": season.get("postseason_rank"),
                        "srs": season.get("srs"),
                        "spOverall": season.get("sp_overall"),
                        "spOffense": season.get("sp_offense"),
                        "spDefense": season.get("sp_defense")
                    })
            
            operations.append(
                pymongo.UpdateOne(
                    {"firstName": coach_doc["firstName"], "lastName": coach_doc["lastName"], "seasons.year": year},
                    {"$set": coach_doc},
                    upsert=True
                )
            )

        result = collection.bulk_write(operations)
        self.log_message(f"Inserted/Updated {result.upserted_count + result.modified_count} coach records for year {year}")

    def extract_and_save_coaches(self):
        """Main method to extract coaches data and save it to the database for the specified years."""
        for year in self.years:
            self.log_message(f"Processing coaches data for year {year}")
            
            coaches_data = self.get_coaches(year)
            if not coaches_data:
                self.log_message(f"No coaches data was found for year {year}.")
                continue

            with tqdm(total=1, desc=f"Processing coaches for {year}", file=sys.stdout) as self.progress_bar:
                self.save_coaches_to_db(coaches_data, year)
                self.progress_bar.update(1)

            self.log_message(f"Processed {len(coaches_data)} coach records for year {year}.")