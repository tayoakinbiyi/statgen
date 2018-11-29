import numpy as np

def D(u,v):
    return(u*np.log(u/v)+(1-u)*np.log((1-u)/(1-v)))

def nCr(n, r):
    lFac=np.log(list(range(1,n+1)))
    cFac=np.cumsum(lFac)
    ans=np.concatenate([[0],cFac[-1]-cFac[0:int(np.ceil(r))]-cFac[[-k for k in range(2,int(np.floor(r))+2)]]])
    return (ans)

def ggStats(d):
    arr=np.array([[np.nan]*j+list(range(d+1-j)) for j in range(1,int(np.ceil(d/2.0))+2)])
    cr=nCr(d,int(np.ceil(d/2.0)))
    return(arr,cr)

def rho(theta):
    if .5<theta<=0.75:
        return(theta-.5)
    else:
        return((1-np.sqrt(1-theta))**2)
