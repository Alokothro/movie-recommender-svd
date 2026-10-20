# movie-recommender-svd

Building a movie recommendation system with linear algebra and the SVD.

**MATH 344L: Applied Linear Algebra Lab**

## About

If we put movie ratings in a matrix, with one row per user and one column per movie, most of the
entries are missing because nobody has seen every movie. A recommender system's job is to fill in those
missing entries. Ratings mostly depend on a few hidden factors (like how much someone likes action
or comedy), so the full matrix should be close to a low-rank matrix. That's what we'll use the SVD for.

## Milestones

| # | Notebook | What it covers |
|---|---|---|
| 1 | [Ratings as a matrix](notebooks/01%20ratings%20matrix.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alokothro/movie-recommender-svd/blob/main/notebooks/01%20ratings%20matrix.ipynb) | A toy ratings matrix, exploring MovieLens 100K, building the 943 × 1682 user-by-movie matrix, and why a low-rank model makes sense |

## Data

We use the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) data set from GroupLens:
100,000 ratings from 943 users on 1,682 movies. The notebook downloads it automatically. The data is
**not** stored in this repository because its license doesn't allow redistribution.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> *ACM Transactions on Interactive Intelligent Systems* 5, 4, Article 19.
> https://doi.org/10.1145/2827872

## Repository layout

```
notebooks/   Colab notebooks, one for each milestone
src/         Reusable Python code for later in the project
figures/     Plots and other saved images
```

## Running the notebooks

The easiest way is the **Open in Colab** button above, then **Runtime → Run all**.

To run them on your own computer instead:

```bash
pip install -r requirements.txt
```

## License

This project is released under the [MIT License](LICENSE).
