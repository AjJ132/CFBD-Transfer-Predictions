from pymongo import MongoClient

from scripts.quarterbacks.clean_database import CleanDatabase
from scripts.quarterbacks.process_quarterbacks import ProcessQuarterbacks
from scripts.quarterbacks.flatten_quarterbacks import FlattenQuarterbacks
from scripts.quarterbacks.get_transfer_information import QuarterbackTransferInformation


def main():
    mongo_client = MongoClient('mongodb://localhost:27017/')

    seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    process_quarterbacks = ProcessQuarterbacks(
        db_client=mongo_client,
        seasons=[2016]
    )

    # print("Processing quarterbacks...")
    process_quarterbacks.process()

    exit()

    print("Flattening quarterback data...")
    flatten_quarterbacks = FlattenQuarterbacks(
        db_client=mongo_client,
        seasons=seasons
    )

    # flatten_quarterbacks.flatten()

    print("Getting transfer information...")
    transfer_information = QuarterbackTransferInformation(
        db_client=mongo_client,
        seasons=seasons
    )

    transfer_information.get_transfer_information()




if __name__ == "__main__":
    main()