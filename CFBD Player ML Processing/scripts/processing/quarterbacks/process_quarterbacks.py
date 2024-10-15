import pymongo
from pymongo import MongoClient
from typing import List, Dict, Any
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json
from bson import json_util

class ProcessQuarterbacks:
    def __init__(self, db_client: MongoClient, seasons: List[int]):
        self.db_client = db_client
        self.seasons = seasons
        self.db = db_client['cfb_data']
        self.geo_locator = Nominatim(user_agent="my_agent")

    def process(self):
        results = []
        for season in self.seasons:
            results.extend(self.process_season(season))
        return results

    def process_season(self, season: int) -> List[Dict[str, Any]]:
        qb_data = []
        players = self.get_quarterbacks_for_season(season)
        for player in players:
            if player['firstName'] == '' or player['lastName'] == '':
                continue
            qb_info = self.compile_quarterback_info(player, season)
            qb_data.append(qb_info)
        return qb_data

    def get_quarterbacks_for_season(self, season: int) -> List[Dict[str, Any]]:
        return list(self.db.players.find({"season": season, "position": "QB"}))
    
    def get_team_by_id(self, team_id: int, season: int) -> Dict[str, Any]:
        return self.db.teams.find_one({"id": team_id, "season": season})

    def compile_quarterback_info(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
        team = self.get_team_by_id(player['team_id'], season)

        if team is None:
            print(f"Team not found for player: {player['id']}")
            exit(1)

        player_info = self.get_stats(player, season)
        print(json.dumps(player_info, indent=4))
        exit(1)

        qb_info = {
            "player": self.get_player_info(player),
            "team": self.get_team_info(player, team),
            "usage": self.get_usage(player, season),
            "stats": self.get_stats(player, season),
            "teamPerformance": self.get_team_performance(player, season),
            "depthChart": self.get_depth_chart(player, season),
            "injuries": self.get_injuries(player, season),
            "disciplinaryActions": self.get_disciplinary_actions(player, season),
            "socialMediaMetrics": self.get_social_media_metrics(player, season),
            "nil": self.get_nil_info(player, season),
            "transferHistory": self.get_transfer_history(player)
        }
        return qb_info

    def get_player_distance_to_home(self, player: Dict[str, Any], team_location_lat: float, team_location_long: float) -> Dict[str, Any]:
        if 'homeLatitude' in player and 'homeLongitude' in player:
            player_location_lat = player['homeLatitude']
            player_location_long = player['homeLongitude']
            distance = geodesic((player_location_lat, player_location_long), (team_location_lat, team_location_long)).miles
            return distance
        
        if 'homeCity' in player and 'homeState' in player:
            location = self.geo_locator.geocode(f"{player['homeCity']}, {player['homeState']}")
            player_location_lat = location.latitude
            player_location_long = location.longitude
            distance = geodesic((player_location_lat, player_location_long), (team_location_lat, team_location_long)).miles
            return distance
        
        return "UNKNOWN"

    def get_player_info(self, player: Dict[str, Any], team: Dict[str, Any]) -> Dict[str, Any]:
        team_location_lat = team['location']['latitude']
        team_location_long = team['location']['longitude']
        player_distance_to_home = self.get_player_distance_to_home(player, team_location_lat, team_location_long)
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
            'distance_to_home': player_distance_to_home
        }
    
    def get_team_info(self, player: Dict[str, Any], team: Dict[str, Any]) -> Dict[str, Any]:
        # Search for the coach using $elemMatch
        coach_element = self.db.coaches.find_one({
            "seasons": {
                "$elemMatch": {
                    "school": player['team'],
                    "year": team['season']
                }
            }
        })
    
        return {
            'conference': team['conference'],
            'coach': f"{coach_element['firstName']} {coach_element['lastName']}" if coach_element else "UNKNOWN"
        }
                    
        from bson import ObjectId
    
    def get_usage(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
        # get player by id and season from player_usage collection
        player_usage = self.db.player_usage.find_one({
            "id": str(player['id']),
            "season": season
        })

        # Convert ObjectId to string for JSON serialization
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
        # Find all playerId by player['id'] and season from player_stats collection
        player_stats_cursor = self.db.player_stats.find({
            "playerId": str(4430841),
            "season": 2023
        })

        # Convert cursor to list of dictionaries
        player_stats_list = list(player_stats_cursor)
        
        # Convert ObjectId to string for each document
        for stat in player_stats_list:
            stat['_id'] = str(stat['_id'])
        
        # Collect passing, rushing, fumbles, and touchdowns categories
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

        #
        passing_stats = {
            'attempts': 0,
            'completions': 0,
            'yards': 0,
            'touchdowns': 0,
            'interceptions': 0,
            'rating': 0
        }

        rushing_stats = {
            'attempts': 0,
            'yards': 0,
            'touchdowns': 0,
            'longRun': 0
        }

        fumbles_stats = {
            'fumbles': 0,
            'lost': 0
        }

        for stat in passing_collections:
            passing_stats['attempts'] += stat['attempts']
            passing_stats['completions'] += stat['completions']
            passing_stats['yards'] += stat['yards']
            passing_stats['touchdowns'] += stat['touchdowns']
            passing_stats['interceptions'] += stat['interceptions']
            passing_stats['rating'] += stat['rating']

        for stat in rushing_collections:
            rushing_stats['attempts'] += stat['attempts']
            rushing_stats['yards'] += stat['yards']
            rushing_stats['touchdowns'] += stat['touchdowns']
            rushing_stats['longRun'] = max(rushing_stats['longRun'], stat['longRun'])


        # Process the stats as needed
        # For now, just return the first stat document if available
        return player_stats_list[0] if player_stats_list else {}


    def get_team_performance(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
        pass

    def get_depth_chart(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
        pass

    def get_injuries(self, player: Dict[str, Any], season: int) -> List[Dict[str, Any]]:
        pass

    def get_disciplinary_actions(self, player: Dict[str, Any], season: int) -> List[Dict[str, Any]]:
        pass

    def get_social_media_metrics(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
        pass

    def get_nil_info(self, player: Dict[str, Any], season: int) -> Dict[str, Any]:
        pass

    def get_transfer_history(self, player: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass