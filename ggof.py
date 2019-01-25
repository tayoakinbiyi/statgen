import numpy as np
import scipy.stats as st
import pdb
import pandas as pd
from scipy.stats import norm
import scipy

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

def qnorm_mu(mu, t, kkk, d):
    return(1 - (norm.cdf(t, loc=mu) - norm.cdf(-t, loc=mu)) - kkk/float(d))

def ggof(z_raw,p,pairwise_cors,arr,cr):
    z = -np.sort(-np.abs(z_raw))
    cor={}
                             
    d=len(z)
    k_vec=np.array(range(d))

    non_zero = np.array(range(int(np.ceil(d/2.0))))
    cor['non_zero']=non_zero    
    cor['non_zero_hc']=non_zero[non_zero>=sum(p<=1.0/d)]

    sigmasq = np.array(range(d),dtype='float')    
    sigmasq[non_zero]=var_st(z[non_zero],d,pairwise_cors)
    
    lam = np.array(range(d),dtype='float')
    rho = np.array(range(d),dtype='float')
    gamma =np.array(range(d),dtype='float')

    lam[non_zero] = 2*norm.sf(z[non_zero])
    rho[non_zero] = (sigmasq[non_zero] - d*lam[non_zero]*(1-lam[non_zero])) / (d*(d-1)*lam[non_zero]*(1-lam[non_zero]))
    gamma[non_zero] = rho[non_zero] / (1-rho[non_zero])

    pq_mat = np.array([[0,0]]*d,dtype='float')
    pq_mat[non_zero, 0] = -lam[non_zero] / (d-1)
    pq_mat[non_zero, 1] = -(1-lam[non_zero]) / (d-1)
    gamma_check = pq_mat.max(1)   

    non_zero = non_zero[gamma[non_zero] >= gamma_check[non_zero]]
    cor['non_zero_ggnull']=non_zero

    if len(non_zero)==0:
        return ({'non_zero':cor['non_zero'],'non_zero_hc':cor['non_zero_hc'],'non_zero_ghc':np.array([]),
                 'non_zero_gbj':np.array([]),'non_zero_ggnull':np.array([]),'ghc':[],'gbj':[],'ggnull':[]})

    param = np.concatenate([k_vec[non_zero].reshape(-1,1),lam[non_zero].reshape(-1,1),gamma[non_zero].reshape(-1,1)],axis=1)
    cor['ggnull'] = -np.apply_along_axis(ebb_gnull,1, param,d=d,ar=arr,cr=cr)
                             
    non_zero_ghc=non_zero[non_zero>=sum(p<=1.0/d)]
    cor['non_zero_ghc']=non_zero_ghc
    cor['ghc']=(k_vec[non_zero_ghc]+1-d*p[non_zero_ghc])/np.sqrt(sigmasq[non_zero_ghc])
    
    muj =np.array([0]*d,dtype='float')
    non_zero_gbj=non_zero[p[non_zero]<(k_vec[non_zero]+1)/float(d)]
    for iii in non_zero_gbj:
        muj[iii] = scipy.optimize.brentq(qnorm_mu, 0, 1000, args = (z[iii],k_vec[iii]+1, d))

    sigmasq_alt = np.array(range(d),dtype='float')   
    sigmasq_alt[non_zero_gbj]=var_st_mu(z[non_zero_gbj], d,mu=muj[non_zero_gbj],pairwise_cors=pairwise_cors)
    
    lam_alt=np.array([0]*d,dtype='float')
    rho_alt = np.array([0]*d,dtype='float')
    gamma_alt = np.array([0]*d,dtype='float')
                               
    lam_alt[non_zero_gbj] = (k_vec[non_zero_gbj]+1) / float(d)
    rho_alt[non_zero_gbj] = (sigmasq_alt[non_zero_gbj] - d*lam_alt[non_zero_gbj]*(1-lam_alt[non_zero_gbj])) / \
         (d*(d-1)*lam_alt[non_zero_gbj]*(1-lam_alt[non_zero_gbj]))
    gamma_alt[non_zero_gbj] = rho_alt[non_zero_gbj] / (1-rho_alt[non_zero_gbj])

    pq_mat_alt =np.array([[0,0]]*d,dtype='float')
    pq_mat_alt[non_zero_gbj, 0] = -lam_alt[non_zero_gbj] / (d-1)
    pq_mat_alt[non_zero_gbj, 1] = -(1-lam_alt[non_zero_gbj]) / (d-1)    
    gamma_check_alt = pq_mat_alt.max(1)
    
    #non_zero_gbj = non_zero_gbj[gamma_alt[non_zero_gbj] >= gamma_check_alt[non_zero_gbj]] 
    
    cor['non_zero_gbj']=non_zero_gbj
    if len(non_zero_gbj)==0:
        cor['gbj']=[]
        return(cor)
    
    null_loglik = np.array([0]*d,dtype='float')
    param = np.concatenate([k_vec[non_zero_gbj].reshape(-1,1),lam[non_zero_gbj].reshape(-1,1),gamma[non_zero_gbj].reshape(-1,1)],axis=1)
    null_loglik[non_zero_gbj] = np.apply_along_axis(ebb_loglik,1,param,d=d,ar=arr,cr=cr)

    alt_loglik = np.array([0]*d,dtype='float')
    alt_param = np.concatenate([k_vec[non_zero_gbj].reshape(-1,1),lam_alt[non_zero_gbj].reshape(-1,1),
        gamma_alt[non_zero_gbj].reshape(-1,1)],axis=1)

    alt_loglik[non_zero_gbj] = np.apply_along_axis(ebb_loglik,1,alt_param,d=d,ar=arr,cr=cr)
    cor['gbj']= alt_loglik[non_zero_gbj] - null_loglik[non_zero_gbj]

    return(cor)

def ebb_gnull(x,d,ar,cr):
    k=int(x[0])
    lam=x[1]
    gamma=x[2]
    return(1-sum(np.exp(cr[0:(k+1)]+np.nansum(np.log(ar[0:(k+1)]*gamma+1-lam),axis=1)+
        np.nansum(np.log(ar.T[0:(k+1)]*gamma+lam),axis=1)-np.nansum(np.log(1+ar[0]*gamma)))))

def ebb_loglik(x, d,ar,cr):
    k=int(x[0])+1
    lam=x[1]
    gamma=x[2]
    try:
        ans=cr[k] + np.nansum(np.log(lam+gamma*ar.T[k])) + np.nansum(np.log(1-lam+gamma*ar[k]))- np.nansum(np.log(1+ar[0]*gamma))
    except:
        pdb.set_trace()
    return(ans)
