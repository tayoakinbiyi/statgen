from scipy.stats import norm
import numpy as np
import pandas as pd

def getRhoBar(pairwise_cors):
    return([np.mean(pairwise_cors), np.mean(pairwise_cors**2), np.mean(pairwise_cors**3), np.mean(pairwise_cors**4),
        np.mean(pairwise_cors**5), np.mean(pairwise_cors**6), np.mean(pairwise_cors**7), np.mean(pairwise_cors**8),
        np.mean(pairwise_cors**9), np.mean(pairwise_cors**10)])
    
def getVarNoMu(t,d,rho):
    odds,evens=getHerm(t,t,rho)  
    ans=d*(2*norm.sf(t)-4*norm.sf(t)**2)+4*d*(d-1)*norm.pdf(t)**2*odds
    return(ans)

def getHerm(t1,t2,rho):
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
    
    odds = ( He1*rho[1]/2 + He3*rho[3]/24 + He5*rho[5]/720 + He7*rho[7]/40320 + He9*rho[9]/3628800 )
    evens = ( He0*rho[0]/1 + He2*rho[2]/6 + He4*rho[4]/120 + He6*rho[6]/5040 + He8*rho[8]/362880 )
    
    return(odds,evens)

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

def vst(binEdges,d,rhoBar):
    z=np.abs(norm.ppf(binEdges/2))
    return(pd.DataFrame({'binEdge':binEdges,'var':getVarNoMu(z,d,rhoBar)},dtype='float32'))                     
