from scipy.stats import norm
import numpy as np

def var_st_mu(t,d,mu,pairwise_cors):
    prob_greater = 1 - ( norm.cdf(t, loc=mu) - norm.cdf(-t, loc=mu) )
    ind_term = d*prob_greater - d*prob_greater**2

    cov_term_gg = d*(d-1)* (norm.sf(t-mu)**2 + norm.pdf(t-mu)**2*np.sum(herm(t-mu,t-mu,pairwise_cors=pairwise_cors),0)) 
    cov_term_ll = d*(d-1)* (1-2*norm.sf(-t-mu)+norm.sf(-t-mu)**2 + norm.pdf(-t-mu)**2*np.sum(herm(-t-mu,-t-mu,
        pairwise_cors=pairwise_cors),0))
    cov_term_diff = d*(d-1)* (norm.sf(t-mu) - norm.sf(t-mu)*norm.sf(-t-mu) - norm.pdf(t-mu)*norm.pdf(-t-mu)*np.sum(herm(t-mu,
        -t-mu,pairwise_cors=pairwise_cors),0))

    cov_term = cov_term_gg + cov_term_ll + 2*cov_term_diff - d*(d-1)*prob_greater**2

    return(cov_term + ind_term)  
    
def var_st(t,d,pairwise_cors):
    odds,evens=herm(t,t,pairwise_cors)  
    ans=d*(2*norm.sf(t)-4*norm.sf(t)**2)+4*d*(d-1)*norm.pdf(t)**2*odds
    return(ans)