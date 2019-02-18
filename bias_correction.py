from statsmodels.distributions.empirical_distribution import ECDF
def quantile_correction(ref, ctr, data, modified = True):
    cdf = ECDF(ctr)
    p = cdf(data)
        
    cor = np.subtract(*[np.nanquantile(x, p) for x in [ref, ctr]])
    mid = np.subtract(*[np.nanquantile(x, 0.5) for x in [ref, ctr]])
    g = np.true_divide(*[np.nanquantile(x, 0.5) for x in [ref, ctr]])
    
    iqr_ref = np.subtract(*np.nanquantile(ref, [0.75, 0.25]))
    iqr_ctr = np.subtract(*np.nanquantile(ctr, [0.75, 0.25])) 
    f = np.true_divide(iqr_ref, iqr_ctr)
    corr = g*mid + f*(cor - mid)
    if modified:
        return data + corr
    else:   
        return data + cor
