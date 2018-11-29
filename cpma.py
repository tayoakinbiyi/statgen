import numpy as np

def exp_lik(x,lam=1,cpma_eps=0.001):
    return(np.exp(-lam*(x-cpma_eps))-np.exp(-lam*(x+cpma_eps)))

def cpma(p_val,cpma_eps=0.001):
    p_val=p_val[(p_val>0)&(p_val<1)]
    x=-np.log(p_val)
    lambda_hat=1.0/np.mean(x)
    return(-2*(sum(np.log(exp_lik(x,1)))-sum(np.log(exp_lik(x,lambda_hat)))))


