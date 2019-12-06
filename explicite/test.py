from resystools.explicite.cfmodel import *
from sklearn.model_selection import train_test_split
import pandas as pd

def run_test_mf():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv('ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1')
    ratings = ratings_base.as_matrix()
    ratings[:, :2] -= 1
    rate_train, rate_test = train_test_split(ratings, test_size=0.33, random_state=42)
    rs = MF(rate_train, K = 2, lam = 0.1, print_every = 2, learning_rate = 2, max_iter = 10, user_based = 0)
    rs.fit()
    RMSE = rs.evaluate_RMSE(rate_test)
    print('\nItem-based MF, RMSE =', RMSE)

def run_test_ib():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

    rate_train = ratings_base.as_matrix()
    rate_test = ratings_test.as_matrix()

        # indices start from 0
    rate_train[:, :2] -= 1
    rate_test[:, :2] -= 1
    rs = ItemBase(rate_train, k = 30)
    rs.fit()
    print("mae ",rs.evaluate_mae(rate_test))
    print("rmse ",rs.evaluate_rmse(rate_test))

def run_test_ub():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

    rate_train = ratings_base.as_matrix()
    rate_test = ratings_test.as_matrix()

        # indices start from 0
    rate_train[:, :2] -= 1
    rate_test[:, :2] -= 1
    rs = UserBase(rate_train, k = 30)
    rs.fit()
    print("mae ",rs.evaluate_mae(rate_test))
    print("rmse ",rs.evaluate_rmse(rate_test))