import numpy as np
import itertools

def estimate_scale_point_sets(src, dst, max_est=50000):
    
    idxs = np.arange(len(src))
    np.random.shuffle(idxs)
    
    # computes cross ratios between all pairs of points
    scales = []
    for i, (j,k) in enumerate(itertools.combinations(idxs, 2)):
        d1 = np.linalg.norm(src[j]-src[k])
        d2 = np.linalg.norm(dst[j]-dst[k])
        scales.append(d2/d1)
        
        if i>max_est:
            break
        
    return np.median(scales)

def procrustes_registration(src, dst):
    """
    Estimates rotation translation and scale of two point sets
    using Procrustes analysis
    
    dst = (src*scale x R.T) + t + residuals
    
    Parameters:
    ----------
    src : numpy.ndarray (N,3)
        transformed points set
    dst : numpy.ndarray (N,3)
        target points set   
        
    Return:
    -------
    scale, rotation matrix, translation and average distance
    between the alligned points sets
    """
    from scipy.linalg import orthogonal_procrustes
    
    assert src.shape[0]==dst.shape[0]
    assert src.shape[1]==dst.shape[1]
    assert src.shape[1]==3    

    P = src.copy()
    Q = dst.copy()

    m1 = np.mean(P, 0) 
    m2 = np.mean(Q, 0)

    P -= m1
    Q -= m2

    norm1 = np.linalg.norm(P)
    norm2 = np.linalg.norm(Q)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    P /= norm1
    Q /= norm2
    R, s = orthogonal_procrustes(Q, P)
    
    scale = s*norm2/norm1
    t = m2-np.dot(m1*scale, R.T)
    
    src_trans = np.dot(src*scale, R.T) + t

    mean_distance = np.linalg.norm(src_trans-dst, axis=1).mean()
    
    return scale, R, t, mean_distance