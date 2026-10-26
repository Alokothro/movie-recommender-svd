# Source

Reusable Python code for the project.

| File | What's in it |
|---|---|
| [evaluate.py](evaluate.py) | `split_ratings` (honest train/test split), `to_matrix` (user-by-movie matrix with `nan` for missing ratings), and `rmse` (error in stars) |
| [models.py](models.py) | `bias_baseline` (overall average + user effect + movie effect) |

To use them from the repository root:

```python
from src.evaluate import split_ratings, to_matrix, rmse
from src.models import bias_baseline
```
