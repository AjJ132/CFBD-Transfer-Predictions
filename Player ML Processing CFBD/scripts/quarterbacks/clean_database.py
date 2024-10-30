import numpy as np
from itertools import combinations
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from rapidfuzz import fuzz
import json
import pandas as pd
from pymongo import MongoClient

class CleanDatabase:
    def __init__(self, db_client: MongoClient):
        self.db_client = db_client
        self.db = self.db_client.get_database('cfb_data_copy')
        self.players_df = None

    def load_data(self):
        # Load players from the database into DataFrame
        players = list(tqdm(self.db.players.find({}), desc="Loading players"))
        print(f"Loaded {len(players)} players.")
        self.players_df = pd.DataFrame(players)

    def compare_pairs(self, pair):
        (i, row), (j, other_row) = pair
        name1 = f"{row['firstName']} {row['lastName']}".lower()
        name2 = f"{other_row['firstName']} {other_row['lastName']}".lower()
        similarity = fuzz.ratio(name1, name2)
        if similarity >= 80 and row['id'] != other_row['id']:
            return {
                'player1': row.to_dict(),
                'player2': other_row.to_dict(),
                'similarity': similarity
            }
        return None

    def process_batch(self, batch_df, similarity_threshold=80):
        """Process a batch of data and return potential duplicates"""
        potential_duplicates = []
        num_rows = len(batch_df)
        if num_rows < 2:
            return potential_duplicates
        
        pairs = combinations(batch_df.iterrows(), 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(tqdm(executor.map(self.compare_pairs, pairs), total=num_rows*(num_rows-1)//2))
        
        potential_duplicates.extend([r for r in results if r])
        return potential_duplicates

    def find_potential_duplicates(self, similarity_threshold=80, batch_size=1000):
        potential_duplicates = []
        num_batches = int(np.ceil(len(self.players_df) / batch_size))

        # Batch process data
        for i in range(num_batches):
            print(f"Processing batch {i+1} of {num_batches}...")
            batch_df = self.players_df.iloc[i * batch_size : (i + 1) * batch_size]
            batch_duplicates = self.process_batch(batch_df, similarity_threshold)
            potential_duplicates.extend(batch_duplicates)

        return potential_duplicates

    def save_to_file(self, potential_duplicates, filename):
        with open(filename, 'w') as f:
            json.dump(potential_duplicates, f, indent=2, default=str)

    def find_duplicates(self):
        self.load_data()
        potential_duplicates = self.find_potential_duplicates()
        self.save_to_file(potential_duplicates, "potential_duplicate_players.json")
        print(f"Found {len(potential_duplicates)} potential duplicate pairs.")
