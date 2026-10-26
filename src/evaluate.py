"""Tools for testing a recommender honestly.

split_ratings hides some ratings for testing, to_matrix turns a table of ratings into a
user-by-movie matrix, and rmse scores predictions in stars.
"""
import numpy as np

N_USERS, N_ITEMS = 943, 1682


def split_ratings(ratings, test_frac=0.10, seed=0):
    """Hide ratings without removing a user's or movie's last training rating.

    Each rating gets about a test_frac chance of being hidden, but a rating is only hidden
    if its user and its movie both still have more than one training rating. Returns
    (train, test) as two DataFrames. The same seed always gives the same split.
    """
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(ratings), dtype=bool)

    user_left = ratings.groupby("user").size().to_dict()
    item_left = ratings.groupby("item").size().to_dict()

    for pos in rng.permutation(len(ratings)):
        row = ratings.iloc[pos]
        user, item = row["user"], row["item"]

        if (rng.random() < test_frac
                and user_left[user] > 1
                and item_left[item] > 1):
            is_test[pos] = True
            user_left[user] -= 1
            item_left[item] -= 1

    return ratings.iloc[~is_test].copy(), ratings.iloc[is_test].copy()


def to_matrix(df):
    """Return a user-by-movie matrix; missing ratings are np.nan.

    User and movie IDs start at 1, so user u and movie i go in row u - 1, column i - 1.
    """
    R = np.full((N_USERS, N_ITEMS), np.nan)
    R[df["user"].to_numpy() - 1,
      df["item"].to_numpy() - 1] = df["rating"].to_numpy()
    return R


def rmse(pred, truth):
    """Root mean squared error, measured in stars."""
    return np.sqrt(np.mean((pred - truth)**2))
