"""Recommender models.

bias_baseline predicts a rating as the overall average plus a user effect plus a movie
effect. It is the baseline the SVD model has to beat.
"""
import numpy as np


def bias_baseline(R):
    """Overall mean plus one user effect and one movie effect.

    R is a user-by-movie matrix with np.nan for missing ratings. Returns (mu, b_user, b_item),
    so the predicted rating for user u and movie i is mu + b_user[u] + b_item[i].
    A movie with no ratings in R gets an effect of 0.
    """
    known = ~np.isnan(R)
    mu = np.nanmean(R)

    b_user = np.nanmean(R - mu, axis=1)
    residual = R - mu - b_user[:, None]

    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])
    has_rating = n_item > 0
    b_item[has_rating] = (
        np.nansum(residual, axis=0)[has_rating] / n_item[has_rating])

    return mu, b_user, b_item
