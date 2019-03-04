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

def herm(t1,t2,pairwise_cors):
    rho_bar1 = np.mean(pairwise_cors)
    rho_bar2 = np.mean(pairwise_cors**2)
    rho_bar3 = np.mean(pairwise_cors**3)
    rho_bar4 = np.mean(pairwise_cors**4)
    rho_bar5 = np.mean(pairwise_cors**5)
    rho_bar6 = np.mean(pairwise_cors**6)
    rho_bar7 = np.mean(pairwise_cors**7)
    rho_bar8 = np.mean(pairwise_cors**8)
    rho_bar9 = np.mean(pairwise_cors**9)
    rho_bar10 = np.mean(pairwise_cors**10)

    He1 = (t1)*(t2)
    He3 = (t1**3 - 3*t1)*(t2**3 - 3*t2)
    He5 = (t1**5 - 10*t1**3 + 15*t1)*(t2**5 - 10*t2**3 + 15*t2)
    He7 = (t1**7 - 21*t1**5 + 105*t1**3 - 105*t1)*(t2**7 - 21*t2**5 + 105*t2**3 - 105*t2)
    He9 = (t1**9 - 36*t1**7 + 378*t1**5 - 1260*t1**3 + 945*t1)*(t2**9 - 36*t2**7 + 378*t2**5 - 1260*t2**3 + 945*t2)
    He0 = 1
    He2 = (t1**2 - 1)*(t2**2 - 1)
    He4 = (t1**4 - 6*t1**2 + 3)*(t2**4 - 6*t2**2 + 3)
    He6 = (t1**6 - 15*t1**4 + 45*t1**2 - 15)*(t2**6 - 15*t2**4 + 45*t2**2 - 15)
    He8 = (t1**8 - 28*t1**6 + 210*t1**4 - 420*t1**2 + 105)*(t2**8 - 28*t2**6 + 210*t2**4 - 420*t2**2 + 105)
    
    odds = ( He1*rho_bar2/2 + He3*rho_bar4/24 + He5*rho_bar6/720 + He7*rho_bar8/40320 + He9*rho_bar10/3628800 )
    evens = ( He0*rho_bar1/1 + He2*rho_bar3/6 + He4*rho_bar5/120 + He6*rho_bar7/5040 + He8*rho_bar9/362880 )
    
    return(odds,evens)