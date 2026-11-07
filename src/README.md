# Source

Reusable Python code for the project.

| File | What's in it |
|---|---|
| [evaluate.py](evaluate.py) | `split_ratings` (honest train/test split), `to_matrix` (user-by-movie matrix with `nan` for missing ratings), and `rmse` (error in stars) |
| [models.py](models.py) | `bias_baseline` (overall average + user effect + movie effect), `baseline_matrix` (the baseline as a full matrix of predictions), `rank_k` (rebuild a rank-k matrix from an SVD), and `iterative_svd_predict` (the iterative SVD model from Milestone 3) |

To use them from the repository root:

```python
from src.evaluate import split_ratings, to_matrix, rmse
from src.models import bias_baseline, baseline_matrix, rank_k, iterative_svd_predict
```
