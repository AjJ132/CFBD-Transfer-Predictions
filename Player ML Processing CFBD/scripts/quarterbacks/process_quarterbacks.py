import pymongo
from pymongo import MongoClient
from typing import List, Dict, Any
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import multiprocessing as mp

class ProcessQuarterbacks:
    def __init__(self, db_client: MongoClient, seasons: List[int], batch_size: int = 100, num_processes: int = 4):
        self.db_client = db_client
        self.seasons = seasons
        self.db = db_client['cfb_data_copy']
        self.batch_size = batch_size
        self.num_processes = num_processes

    def process(self):
        pool = mp.Pool(processes=self.num_processes)
        all_qb_infos = []

        for season in self.seasons:
            players = self.get_quarterbacks_for_season(season)
            chunk_size = max(1, len(players) // self.num_processes)
            player_chunks = [players[i:i+chunk_size] for i in range(0, len(players), chunk_size)]

            results = []
            for chunk in player_chunks:
                result = pool.apply_async(self.process_player_chunk, args=(chunk, season))
                results.append(result)

            for result in tqdm(results, desc=f"Processing QBs for season {season}"):
                all_qb_infos.extend(result.get())

        pool.close()
        pool.join()

        self.save_player_profiles_to_db(all_qb_infos)

    @staticmethod
    def process_player_chunk(players: List[Dict[str, Any]], season: int) -> List[Dict[str, Any]]:
        processor = ProcessQuarterbacks.PlayerProcessor()
        return processor.process_players(players, season)

    class PlayerProcessor:
        def __init__(self):
            self.geo_locator = Nominatim(user_agent="my_agent")
            self.db_client = MongoClient()  # Assuming default connection settings
            self.db = self.db_client['cfb_data_copy']

        def process_players(self, players: List[Dict[str, Any]], season: int) -> List[Dict[str, Any]]:
            qb_infos = []
            for player in players:
                qb_info = self.compile_quarterback_info(player, season)
                if qb_info:
                    qb_infos.append(qb_info)
            return qb_infos

        def compile_quarterback_info(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
            team = self.get_team_by_id(player['team_id'], season)

            if team is None:
                print(f"Team not found for player: {player['id']}")
                return None

            qb_info = {
                "player": self.get_player_info(player, team),
                "team": self.get_team_info(player, team),
                "usage": self.get_usage(player, season),
                "stats": self.get_stats(player, season),
                "teamPerformance": self.get_team_performance(player, season),
                "conferenceLevel": self.set_conference_level(team),
                "transferRiskFactor": self.set_transfer_risk_factor(player)
            }

            return qb_info

        def get_team_by_id(self, team_id: int, season: int) -> Dict[str, Any]:
            return self.db.teams.find_one({"id": team_id, "season": season})

        def get_player_info(self, player: Dict[str, Any], team: Dict[str, Any]) -> Dict[str, Any]:
            team_location = team['location']
            return {
                'playerId': player['id'],
                'firstName': player['firstName'],
                'lastName': player['lastName'],
                'position': player['position'],
                'teamId': player['team_id'],
                'teamName': player['team'],
                'height': player['height'],
                'weight': player['weight'],
                'class': player['year'],
                'distance_to_home': self.get_player_distance_to_home(player, team_location['latitude'], team_location['longitude'])
            }

        def get_player_distance_to_home(self, player: Dict[str, Any], team_location_lat: float, team_location_long: float) -> float:
            if 'homeLatitude' in player and 'homeLongitude' in player:
                return geodesic((player['homeLatitude'], player['homeLongitude']), (team_location_lat, team_location_long)).miles
            
            if 'homeCity' in player and 'homeState' in player:
                location = self.geo_locator.geocode(f"{player['homeCity']}, {player['homeState']}")
                if location:
                    return geodesic((location.latitude, location.longitude), (team_location_lat, team_location_long)).miles
            
            return -1

        def get_team_info(self, player: Dict[str, Any], team: Dict[str, Any]) -> Dict[str, Any]:
            coach = self.db.coaches.find_one({
                "seasons": {
                    "$elemMatch": {
                        "school": player['team'],
                        "year": team['season']
                    }
                }
            })
            return {
                'conference': team['conference'],
                'coach': f"{coach['firstName']} {coach['lastName']}" if coach else "UNKNOWN"
            }

        def get_usage(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
            player_usage = self.db.player_usage.find_one({
                "id": str(player['id']),
                "season": season
            })

            if player_usage is not None:
                player_usage['_id'] = str(player_usage['_id'])
        
            usage = player_usage['usage'] if player_usage is not None else None
        
            if player_usage is None or usage is None:
                return {
                    'passingDowns': -999,
                    'standardDowns': -999,
                    'thirdDown': -999,
                    'secondDown': -999,
                    'firstDown': -999,
                    'rush': -999,
                    'pass': -999,
                    'overall': -999
                }
            
            return {
                'passingDowns': usage['passingDowns'],
                'standardDowns': usage['standardDowns'],
                'thirdDown': usage['thirdDown'],
                'secondDown': usage['secondDown'],
                'firstDown': usage['firstDown'],
                'rush': usage['rush'],
                'pass': usage['pass'],
                'overall': usage['overall']
            }

        def get_stats(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
            player_stats_cursor = self.db.player_stats.find({
                "playerId": str(player['id']),
                "season": season
            })

            player_stats_list = list(player_stats_cursor)
            
            for stat in player_stats_list:
                stat['_id'] = str(stat['_id'])
            
            passing_collections = []
            rushing_collections = []
            fumbles_collections = []

            for stat in player_stats_list:
                if stat['category'] == 'passing':
                    passing_collections.append(stat)
                elif stat['category'] == 'rushing':
                    rushing_collections.append(stat)
                elif stat['category'] == 'fumbles':
                    fumbles_collections.append(stat)

            passing_stats = {
                'attempts': 0,
                'completions': 0,
                'yards': 0,
                'touchdowns': 0,
                'interceptions': 0,
                'pct': 0
            }

            rushing_stats = {
                'ypc': 0,
                'touchdowns': 0,
                'carries': 0,
                'yards': 0,
                'long': 0
            }

            fumbles_stats = {
                'fumbles': 0,
                'recovered': 0,
                'lost': 0
            }

            for stat in passing_collections:
                if stat['statType'] == 'ATT':
                    passing_stats['attempts'] += stat['stat']
                elif stat['statType'] == 'COMPLETIONS':
                    passing_stats['completions'] += stat['stat']
                elif stat['statType'] == 'YDS':
                    passing_stats['yards'] += stat['stat']
                elif stat['statType'] == 'TD':
                    passing_stats['touchdowns'] += stat['stat']
                elif stat['statType'] == 'INT':
                    passing_stats['interceptions'] += stat['stat']
                elif stat['statType'] == 'PCT':
                    passing_stats['pct'] += stat['stat']

            for stat in rushing_collections:
                if stat['statType'] == 'YPC':
                    rushing_stats['ypc'] += stat['stat']
                elif stat['statType'] == 'TD':
                    rushing_stats['touchdowns'] += stat['stat']
                elif stat['statType'] == 'CAR':
                    rushing_stats['carries'] += stat['stat']
                elif stat['statType'] == 'YDS':
                    rushing_stats['yards'] += stat['stat']
                elif stat['statType'] == 'LONG':
                    rushing_stats['long'] += stat['stat']

            for stat in fumbles_collections:
                if stat['statType'] == 'FUM':
                    fumbles_stats['fumbles'] += stat['stat']
                elif stat['statType'] == 'REC':
                    fumbles_stats['recovered'] += stat['stat']
                elif stat['statType'] == 'LOST':
                    fumbles_stats['lost'] += stat['stat']

            return {
                'passing': passing_stats,
                'rushing': rushing_stats,
                'fumbles': fumbles_stats
            }

        def get_team_performance(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
            # Implement team performance retrieval logic here
            return {}

        def set_conference_level(self, team: Dict[str, Any]) -> str:
            if team['conference'] in ['SEC', 'ACC', 'Big Ten', 'Big 12', 'Pac-12']:
                return 'P5'
            else:
                return 'G5'

        def set_transfer_risk_factor(self, player: Dict[str, Any]) -> float:
            # switch statement on year
            #values are calculated manually based on historical data
            match player['year']:
                case 1:
                    return 0.0542
                case 2:
                    return 0.1006
                case 3:
                    return 0.0932
                case 4:
                    return 0.1307
                case _:  # default
                    return 0.0
                
            

    def get_quarterbacks_for_season(self, season: int) -> List[Dict[str, Any]]:
        return list(self.db.players.find({"season": season, "position": "QB"}))

    def save_player_profiles_to_db(self, profiles: List[Dict[str, Any]]) -> None:
        for i in range(0, len(profiles), self.batch_size):
            batch = profiles[i:i+self.batch_size]
            self.db.quarterback_profiles.insert_many(batch)

