import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from scipy.spatial.distance import hamming as scipy_hamming # type: ignore

RSA_METRICS = ['pearsonr', 'cosine', 'mse', 'euclidean', 'hamming']

def _as_sample_matrix(x: np.ndarray) -> np.ndarray:
    """
    Convert input to 2D array where rows are samples and columns are features.
    Args:
        x: array-like of shape (n_samples, n_features) or (n_features,)

    Returns:
        2D array of shape (n_samples, n_features)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        return x.reshape(1, 1)

    if x.ndim == 1:
        return x.reshape(-1, 1)

    return x.reshape(x.shape[0], -1)

def pearsonr(x:np.ndarray, y:np.ndarray=None) -> np.ndarray:
    """
    Compute Pearson correlation between rows of x and y. If y is None, compute between rows of x.
    
    Args:
        x: (n_samples, n_features) array
        y: (m_samples, n_features) array or None
        
    Returns:
        (n_samples, m_samples) array of Pearson correlations"""

    xs = _as_sample_matrix(x)
    xs = xs - xs.mean(axis=1, keepdims=True)
    xs = xs / (xs.std(axis=1, keepdims=True) + 1e-10)

    if y is not None:
        ys = _as_sample_matrix(y)
        ys = ys - ys.mean(axis=1, keepdims=True)
        ys = ys / (ys.std(axis=1, keepdims=True) + 1e-10)

        assert xs.shape[1] == ys.shape[1]
        corr = (xs @ ys.T) / xs.shape[1]
    else:
        corr = (xs @ xs.T) / xs.shape[1]
    return np.nan_to_num(corr, nan=2.0, posinf=2.0, neginf=2.0)

def cosine(x:np.ndarray, y:np.ndarray=None) -> np.ndarray:
    """Compute cosine similarity between rows of x and y. If y is None, compute between rows of x.

    Args:
        x: (n_samples, n_features) array
        y: (m_samples, n_features) array or None

    Returns:
        (n_samples, m_samples) array of cosine similarities
    """

    x = _as_sample_matrix(x)

    if y is not None:
        y = _as_sample_matrix(y)
        cos_sim = cosine_similarity(x, y)
        return cos_sim
    else:
        cos_sim = cosine_similarity(x)
        return cos_sim

def mse(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Compute mean squared error between rows of x and y. If y is None, compute between rows of x.
    
    Args:
        x: (n_samples, n_features) array
        y: (m_samples, n_features) array or None

    Returns:
        (n_samples, m_samples) array of mean squared errors
    """

    x = _as_sample_matrix(x)
    if y is None:
        y = x
    else:
        y = _as_sample_matrix(y)
        assert x.shape[1] == y.shape[1], "Feature dims must match"

    d = x.shape[1]

    x2 = np.mean(x**2, axis=1, keepdims=True)       
    y2 = np.mean(y**2, axis=1, keepdims=True)        
    xy = (x @ y.T) / d                               

    mse = x2 + y2.T - 2 * xy                      
    return mse

def euclidean(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Compute Euclidean distance between rows of x and y. If y is None, compute between rows of x.
    
    Args:
        x: (n_samples, n_features) array
        y: (m_samples, n_features) array or None
        
    Returns:    
        (n_samples, m_samples) array of Euclidean distances
    """

    x = _as_sample_matrix(x)
    if y is None:
        y = x
    else:
        y = _as_sample_matrix(y)
        assert x.shape[1] == y.shape[1], "Feature dims must match"

    x2 = np.sum(x**2, axis=1, keepdims=True)       
    y2 = np.sum(y**2, axis=1, keepdims=True)        
    xy = (x @ y.T)                               

    euclidean = np.sqrt(x2 + y2.T - 2 * xy)                      
    return euclidean

def hamming(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Compute Hamming distance between rows of x and y. If y is None, compute between rows of x.
    
    Args:
        x: (n_samples, n_features) array
        y: (m_samples, n_features) array or None
        
    Returns:
        (n_samples, m_samples) array of Hamming distances
    """

    x = _as_sample_matrix(x)
    if y is None:
        y = x
    else:
        y = _as_sample_matrix(y)
        assert x.shape[1] == y.shape[1], "Feature dims must match"

    return np.mean(x[:, None, :] != y[None, :, :], axis=-1)

