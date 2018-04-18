import os
os.environ["OMP_NUM_THREADS"] = "10"

import sys
import pandas as pd
import numpy as np
import turicreate as tc


for i in range(1, 53):
    print("running batch %d" % i)
    batch = pd.read_csv("batches/batch_%d_train.dat" % i)
    test_users = pd.read_csv("batches/batch_%d_test.dat" % i)
    model = tc.ranking_factorization_recommender.create(tc.SFrame(batch), 'user', 'item', verbose=False)
    results = model.recommend(users=test_users.user.values, k=100, exclude_known=True, verbose=False)
    results.to_dataframe()[['user','item','rank']].to_csv('batches/batch_%d_predictions.dat', sep=' ', header=False)
