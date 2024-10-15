from pymongo import MongoClient

from scripts.processing.quarterbacks.process_quarterbacks import ProcessQuarterbacks


def main():
    mongo_client = MongoClient('mongodb://localhost:27017/')

    seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    process_quarterbacks = ProcessQuarterbacks(
        db_client=mongo_client,
        seasons=seasons
    )

    print("Processing quarterbacks...")
    process_quarterbacks.process()




if __name__ == "__main__":
    main()