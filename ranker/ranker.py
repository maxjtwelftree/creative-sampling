import numpy as np
import pandas as pd
from compute_elo import Elo
import random

class Ranker:
    def __init__(self, elo, df=None):
        self.elo = elo
        if df is not None:
            self.df = df
    def select(self):
        pair = random.sample(range(len(self.df)), 2)
        return pair
    
    def update(self, samples, winner):
        p1_id = self.df.iloc[samples[0]]["id"]
        p2_id = self.df.iloc[samples[1]]["id"]
        p1 = self.df.loc[self.df["id"] == p1_id, "rating"].iloc[0]
        p2 = self.df.loc[self.df["id"] == p2_id, "rating"].iloc[0]

        print(f"p1 {p1} p2 {p2}")

        updates = self.elo.rank(p1, p2, winner)

        print("updates")
        print(updates)

        self.df.loc[self.df["id"] == p1_id, "rating"] = updates[0]
        self.df.loc[self.df["id"] == p2_id, "rating"] = updates[1]

        return
    
    def determine_outcome(self, indices):
        return 0
    
    def run_update(self):
        samples = self.select()
        winner = self.determine_outcome(samples)
        self.update(samples, winner)