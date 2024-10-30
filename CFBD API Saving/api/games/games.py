import pymongo
import cfbd 
from cfbd.models.game import Game
from pymongo import MongoClient
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import threading
import sys
from datetime import datetime

class CFBGameExtractor:
    def __init__(self, api_client: cfbd.ApiClient, db_client: MongoClient, years: List[int], 
                 max_retries: int = 3, base_wait: float = 1.0, batch_size: int = 1000):
        self.api_client = api_client
        self.db_client = db_client
        self.games_api = cfbd.GamesApi(self.api_client)
        self.years = years
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.log_lock = threading.Lock()
        self.progress_bar = None
        self.batch_size = batch_size
        
        # Initialize database connections with write concern configurations
        self.db = self.db_client.get_database(
            "cfb_data",
            write_concern=pymongo.WriteConcern(w=1, j=False)  # Acknowledge writes but don't wait for journal
        )
        
        # Create indexes for faster upserts
        self._ensure_indexes()
        
        # Initialize bulk operation buffers
        self.games_buffer = []
        self.performances_buffer = []
        self.buffer_lock = threading.Lock()
        self.target_game_id = 401628374  # Add this line to track specific game

    def log_message(self, message: str):
        """Thread-safe logging function that prints below the progress bar."""
        with self.log_lock:
            if self.progress_bar:
                current_position = self.progress_bar.n
                self.progress_bar.clear()
                print(message)
                self.progress_bar.update(0)
                self.progress_bar.refresh()

    # Replace all other log_message calls with print
    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                # Debug the result type
                return result
            except cfbd.ApiException as e:
                if e.status == 429 and attempt < self.max_retries - 1:
                    wait_time = self.base_wait * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

    def get_games_by_week(self, year: int, week: int) -> List[Dict[str, Any]]:
        """Fetch games from the CFB API for a specific year and week and return as a list of dictionaries."""
        try:
            print(f"\n[DEBUG] Attempting to fetch games for Year {year} Week {week}")
            
            games = self.retry_with_backoff(self.games_api.get_games, year=year, week=week, classification="fbs")
            
            print(f"[DEBUG] Number of games received: {len(games) if games else 0}")
            
            processed_games = []
            
            for game in games:
                try:
                    # Convert game to dictionary and handle None values
                    game_dict = {}
                    
                    # Extract all attributes, handling None values
                    for attr in dir(game):
                        if not attr.startswith('_'):  # Skip private attributes
                            value = getattr(game, attr)
                            if isinstance(value, (datetime, cfbd.models.season_type.SeasonType, 
                                            cfbd.models.division_classification.DivisionClassification)):
                                # Convert special types to strings
                                game_dict[attr] = str(value)
                            else:
                                game_dict[attr] = value

                    # Handle None values for line scores
                    game_dict['home_line_scores'] = ([0] * 4 if game_dict.get('home_line_scores') is None 
                                                else game_dict.get('home_line_scores', []))
                    game_dict['away_line_scores'] = ([0] * 4 if game_dict.get('away_line_scores') is None 
                                                else game_dict.get('away_line_scores', []))
                    
                    # Ensure line scores have 4 elements
                    while len(game_dict['home_line_scores']) < 4:
                        game_dict['home_line_scores'].append(0)
                    while len(game_dict['away_line_scores']) < 4:
                        game_dict['away_line_scores'].append(0)
                    
                    # Handle other None values
                    game_dict['home_points'] = game_dict.get('home_points', 0)
                    game_dict['away_points'] = game_dict.get('away_points', 0)
                    game_dict['completed'] = game_dict.get('completed', False)
                    
                    # Debug target game
                    if game_dict.get('id') == self.target_game_id:
                        print(f"\n[RAW DATA] Found target game in API response:")
                        print(f"Raw game data: {game_dict}\n")
                    
                    processed_games.append(game_dict)
                    
                except Exception as e:
                    print(f"Error processing individual game: {str(e)}")
                    print(f"Problematic game data: {game}")
                    print(f"Full error details:", str(e))
                    continue
            
            print(f"[DEBUG] Successfully processed {len(processed_games)} games for Year {year} Week {week}")
            return processed_games
            
        except cfbd.ApiException as e:
            print(f"API error while fetching games for year {year} and week {week}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error fetching games for year {year} and week {week}: {e}")
            print(f"Full error details: {str(e)}")
            return []
    
    def get_player_game_performance(self, game_id: int) -> List[Dict[str, Any]]:
        """Fetch player game performance from the CFB API for a specific game and return as a list of dictionaries."""
        try:
            performances = self.retry_with_backoff(self.games_api.get_game_player_stats, id=game_id)
            return [performance.to_dict() for performance in performances]
        except cfbd.ApiException as e:
            self.log_message(f"An error occurred while fetching player game performance for game {game_id}: {e}")
            return []

    def process_player_game_performance(self, game: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process player game performance data for a single game.
        
        Args:
            game: Dictionary containing game data with nested team and player statistics
            
        Returns:
            List of dictionaries where each dictionary contains all stats for a single player
        """
        player_stats = {}

        game = game[0]

        # Process each team's stats
        for team in game["teams"]:
            team_name = team["team"]
            
            # Process each statistical category (passing, rushing, etc)
            for category in team.get("categories", []):
                category_name = category["name"]
                
                # Process each stat type within the category
                for stat_type in category.get("types", []):
                    stat_name = stat_type["name"]
                    
                    # Process each athlete's stats
                    for athlete in stat_type.get("athletes", []):
                        player_id = athlete["id"]
                        
                        # Initialize player entry if not exists
                        if player_id not in player_stats:
                            player_stats[player_id] = {
                                "player_id": player_id,
                                "name": athlete["name"],
                                "team": team_name
                            }
                        
                        # Add the stat using category_name and stat_name as keys
                        stat_key = f"{category_name}_{stat_name.lower()}"
                        player_stats[player_id][stat_key] = athlete["stat"]
        
        # Convert dictionary to list
        return list(player_stats.values())
    
    def _process_week(self, year: int, week: int):
        """Process a single week's worth of games."""
        try:
            games = self.get_games_by_week(year, week)
            if not games:
                return

            self.save_games_to_db(games, year, week)
            
            # Process games in parallel with a smaller thread pool
            with ThreadPoolExecutor(max_workers=3) as game_executor:
                game_futures = [
                    game_executor.submit(self.process_single_game, game['id'])
                    for game in games
                ]
                
                performances = []
                for future in as_completed(game_futures):
                    try:
                        result = future.result()
                        if result:
                            performances.extend(result)
                    except Exception as e:
                        self.log_message(f"Error processing game: {str(e)}")
                
                if performances:
                    self.save_player_performances_to_db(performances, year, week)
                    
        except Exception as e:
            self.log_message(f"Error processing week {week} for year {year}: {str(e)}")


    def convert_datetime_to_str(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to strings in a dictionary."""
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, dict):
                data[key] = self.convert_datetime_to_str(value)
            elif isinstance(value, list):
                data[key] = [
                    self.convert_datetime_to_str(item) if isinstance(item, dict) 
                    else str(item) if isinstance(item, datetime)
                    else item 
                    for item in value
                ]
            elif value is None:
                data[key] = None  # Preserve None values
        return data
    
    def _ensure_indexes(self):
        """Create necessary indexes for better performance."""
        self.db.games.create_index([
            ("id", pymongo.ASCENDING),
            ("season", pymongo.ASCENDING),
            ("week", pymongo.ASCENDING)
        ], unique=True)
        
        self.db.player_performances.create_index([
            ("game_id", pymongo.ASCENDING),
            ("player_id", pymongo.ASCENDING),
            ("season", pymongo.ASCENDING),
            ("week", pymongo.ASCENDING)
        ], unique=True)

    def _flush_buffer(self, collection_name: str, buffer: List[Dict]):
        """Flush a buffer to the database using bulk operations."""
        if not buffer:
            return

        collection = self.db[collection_name]
        try:
            result = collection.bulk_write(buffer, ordered=False)
            
            # Debug for target game after DB operation
            if collection_name == "games":
                # Verify if target game is in database after flush
                target_game = collection.find_one({"id": self.target_game_id})
                if target_game:
                    print(f"\n[DB CONFIRMATION] Target game found in database after flush:")
                    print(f"Stored data: {target_game}\n")
                else:
                    print(f"\n[DB WARNING] Target game not found in database after flush\n")
            
            buffer.clear()
        except pymongo.errors.BulkWriteError as e:
            # Log only actual errors, not duplicate key errors
            real_errors = [err for err in e.details['writeErrors'] 
                         if err['code'] != 11000]  # 11000 is duplicate key error
            if real_errors:
                print(f"Bulk write errors in {collection_name}: {real_errors}")
            
            # Check if target game was affected by any errors
            target_game_errors = [err for err in e.details['writeErrors'] 
                                if err.get('op', {}).get('q', {}).get('id') == self.target_game_id]
            if target_game_errors:
                print(f"\n[DB ERROR] Errors affecting target game: {target_game_errors}\n")
            
            buffer.clear()

    def save_games_to_db(self, games: List[Dict[str, Any]], season: int, week: int):
        """Buffer game data for bulk insertion."""
        if not games:
            print(f"[DEBUG] No games to save for season {season} week {week}")
            return
            
        print(f"[DEBUG] Attempting to save {len(games)} games for season {season} week {week}")
        
        operations = []
        for game in games:
            if not isinstance(game, dict):
                print(f"[WARNING] Invalid game data type: {type(game)}")
                continue
                
            # Debug for target game before DB operation
            if game.get('id') == self.target_game_id:
                print(f"\n[DB OPERATION] Preparing to save target game to database:")
                print(f"Game data being saved: {game}\n")
            
            # Create a sanitized version of the game data
            game_data = {
                "id": game.get("id"),
                "season": season,
                "week": week,
                "homeTeam": game.get("home_team", "Unknown"),
                "awayTeam": game.get("away_team", "Unknown"),
                "homePoints": game.get("home_points", 0),
                "awayPoints": game.get("away_points", 0),
                "homeLineScores": game.get("home_line_scores", [0, 0, 0, 0]),
                "awayLineScores": game.get("away_line_scores", [0, 0, 0, 0]),
                "startDate": game.get("start_date"),
                "venue": game.get("venue"),
                "completed": game.get("completed", False),
                "conferenceGame": game.get("conference_game", False),
                "neutralSite": game.get("neutral_site", False),
                "venueId": game.get("venue_id"),
                "homeConference": game.get("home_conference"),
                "awayConference": game.get("away_conference"),
            }
            
            operations.append(
                pymongo.UpdateOne(
                    {"id": game["id"], "season": season, "week": week},
                    {"$set": game_data},
                    upsert=True
                )
            )

        with self.buffer_lock:
            self.games_buffer.extend(operations)
            if len(self.games_buffer) >= self.batch_size:
                try:
                    print(f"[DEBUG] Flushing buffer with {len(self.games_buffer)} operations")
                    self._flush_buffer("games", self.games_buffer)
                except Exception as e:
                    print(f"Error during database flush: {str(e)}")
    
    def save_player_performances_to_db(self, performances: List[Dict[str, Any]], 
                                     season: int, week: int):
        """Buffer player performance data for bulk insertion."""
        operations = [
            pymongo.UpdateOne(
                {
                    "game_id": perf["game_id"],
                    "player_id": perf["player_id"],
                    "season": season,
                    "week": week
                },
                {"$set": {**perf, "season": season, "week": week}},
                upsert=True
            ) for perf in performances
        ]

        with self.buffer_lock:
            self.performances_buffer.extend(operations)
            if len(self.performances_buffer) >= self.batch_size:
                self._flush_buffer("player_performances", self.performances_buffer)

    def process_single_game(self, game_id: int) -> List[Dict[str, Any]]:
        """Process a single game's player performance data with error handling."""
        try:
            raw_game_performance = self.get_player_game_performance(game_id)
            if not raw_game_performance:
                self.log_message(f"No performance data found for game {game_id}")
                return []
                
            processed_performance = self.process_player_game_performance(raw_game_performance)
            
            # Add game_id to each player's performance record
            for performance in processed_performance:
                performance['game_id'] = game_id
                
            return processed_performance
            
        except Exception as e:
            self.log_message(f"Error processing game {game_id}: {str(e)}")
            return []

    def extract_and_save_games(self):
        """Extract and save game data with optimized parallel processing."""
        try:
            for year in self.years:
                self.log_message(f"Processing year {year}")
                
                # Process all weeks for the year in parallel
                with ThreadPoolExecutor(max_workers=5) as year_executor:
                    week_futures = {
                        year_executor.submit(self._process_week, year, week): (year, week)
                        for week in range(1, 16)
                    }
                    
                    for future in as_completed(week_futures):
                        year, week = week_futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            self.log_message(f"Failed to process {year} Week {week}: {str(e)}")
                
                # Flush any remaining data in buffers
                with self.buffer_lock:
                    self._flush_buffer("games", self.games_buffer)
                    self._flush_buffer("player_performances", self.performances_buffer)
                
        finally:
            # Ensure final flush of any remaining data
            with self.buffer_lock:
                self._flush_buffer("games", self.games_buffer)
                self._flush_buffer("player_performances", self.performances_buffer)

    




                