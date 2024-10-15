from api.teams.teams import CFBTeamExtractor
from api.players.players import CFBPlayerUsageExtractor 
from api.coaches.coaches import CFBCoachesExtractor
from api.stats.stats import CFBPlayerStatsExtractor
import cfbd
from pymongo import MongoClient

def main():
    configuration = cfbd.Configuration(
        access_token='E2ljpt8DSsqVj9JvHpn7IJrx7pw0sdsNTHFfM30JUCMWDyHgQra5yWs8U3twu+ZN'
    )

    seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    # seasons = [2023]

    with cfbd.ApiClient(configuration) as api_client:
        mongo_client = MongoClient('mongodb://localhost:27017/')
        
        # Extract and save team data
        team_extractor = CFBTeamExtractor(
            api_client=api_client,
            db_client=mongo_client,
            years=seasons
        )
        # team_extractor.extract_and_save_teams()

        # Extract and save player usage data
        player_usage_extractor = CFBPlayerUsageExtractor(
            api_client=api_client,
            db_client=mongo_client,
            years=seasons
        )
        # player_usage_extractor.extract_and_save_player_usage()

        #extract coaches information
        coaches_extractor = CFBCoachesExtractor(
            api_client=api_client,
            db_client=mongo_client,
            years=seasons
        )

        # coaches_extractor.extract_and_save_coaches()

        # Extract and save player stats
        stats_extractor = CFBPlayerStatsExtractor(
            api_client=api_client,
            db_client=mongo_client,
            years=seasons
        )

        stats_extractor.extract_and_save_player_stats()



if __name__ == '__main__':
    main()