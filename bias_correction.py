import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

# data required reference to correct with respect to, control data for the overlapping period

# modified quantile correction based on Bai et. al. (2016) 
def modified_quantile_correction(ref, ctr, data):
    cdf = ECDF(ctr)
    p = cdf(data)
        
    cor = np.subtract(*np.nanquantile([ref, ctr], p))
    mid = np.subtract(*np.nanquantile([ref, ctr], 50))
    
    iqr_ref = np.subtract(*np.nanquantile(ref, [75, 25]))
    iqr_ctr = np.subtract(*np.nanquantile(ctr, [75, 25]))
    
    g = np.true_divide(*np.nanquantile([ref, ctr], 50))
    f = np.true_divide(iqr_ref, iqr_ctr)
    
    correction = g*mid + f*(cor - mid)
    return data + correction
    

