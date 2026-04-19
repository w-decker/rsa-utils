import numpy as np # type: ignore
from scipy.stats import spearmanr # type: ignore

from .metrics import (
    RSA_METRICS,
    pearsonr,
    cosine,
    mse,
    euclidean,
    hamming
)

METRIC_FUNCTIONS = {
    'pearsonr': pearsonr,
    'cosine': cosine,
    'mse': mse,
    'euclidean': euclidean,
    'hamming': hamming,
}

def rdm(x:np.ndarray, metric:str) -> np.ndarray:
    """Compute representational dissimilarity matrix (RDM) for input x using specified metric.
    
    Args:
        x: (n_samples, n_features) array
        metric: one of 'pearsonr', 'cosine', 'mse', 'euclidean', 'hamming'
        
    Returns:
        (n_samples, n_samples) RDM array"""

    assert metric in RSA_METRICS, f"Metric {metric} not supported. Choose from {RSA_METRICS}."

    return 1 - METRIC_FUNCTIONS[metric](x)
    
def rsa(x1:np.ndarray, x2:np.ndarray, metric:str) -> float:
    """Compute representational similarity analysis (RSA) between two sets of representations x1 and x2 using specified metric.

    Args:
        x1: (n_samples, n_features) array
        x2: (m_samples, n_features) array
        metric: one of 'pearsonr', 'cosine', 'mse', 'euclidean', 'hamming'

    Returns:
        Spearman rank correlation between upper triangles of RDMs for x1 and x2    
    """

    rdm1 = rdm(x1, metric)
    rdm2 = rdm(x2, metric)

    # vectorize upper triangle
    iu = np.triu_indices(rdm1.shape[0], k=1) # see Ritchie et al. (2017) for why not to include diagonal
    vec1 = rdm1[iu]
    vec2 = rdm2[iu]

    # compute spearman rank correlation
    rho, pval = spearmanr(vec1, vec2)
    return rho, pval