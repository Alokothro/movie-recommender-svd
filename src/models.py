"""Recommender models.

bias_baseline predicts a rating as the overall average plus a user effect plus a movie
effect. It is the baseline the SVD models have to beat. baseline_matrix turns it into a full
matrix of predictions, rank_k rebuilds a low-rank matrix from an SVD, and
iterative_svd_predict is the iterative SVD model from Milestone 3.
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


def baseline_matrix(R):
    """Full matrix: overall mean plus user and movie effects.

    Entry (u, i) is mu + b_user[u] + b_item[i] from bias_baseline(R), so every user and
    movie gets a prediction, including the ones missing from R.
    """
    mu, b_user, b_item = bias_baseline(R)
    return mu + b_user[:, None] + b_item[None, :]


def rank_k(U, s, Vt, k):
    """Build a rank-k matrix from an already computed SVD.

    Keeps the first k singular values and their matching columns of U and rows of Vt.
    """
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def iterative_svd_predict(R, k, n_iter=10):
    """Predict ratings by repeatedly imputing missing deviations.

    Starts from the deviations R - baseline, with 0 in every gap. Each round takes an SVD,
    keeps the rank-k part, and uses it to refill the gaps while the known deviations stay
    fixed. Returns a full matrix of predicted ratings (the last rank-k deviations plus the
    baseline). n_iter must be at least 1.
    """
    known = ~np.isnan(R)
    B = baseline_matrix(R)
    X = np.where(known, R - B, 0.0)

    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = rank_k(U, s, Vt, k)
        X = np.where(known, R - B, L)

    return L + B
