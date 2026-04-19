# rsa-utils

Quick utils for representational simliarity analysis (RSA) for scalar or array-based features.

# Installation

```
pip install git+https://github.com/w-decker/rsa-utils.git
```

# Usage

```python
import numpy as np
from rsa_utils import rsa

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

rho, pval = rsa(x, y, metric='euclidean')
print(f"Spearman r: {rho}, p-value: {pval}")
```
```text
Spearman r: 1.0, p-value: 0.0
```

## Additional metrics

```python
from rsa_utils.metrics import RSA_METRICS

for metric in RSA_METRICS:
    print(f'{metric}')
```
```text
pearsonr
cosine
mse
euclidean
hamming
```