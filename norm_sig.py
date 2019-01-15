import scipy.stats as st
import numpy as np
import pdb

def norm_sig(N,cov,neg_pct):
    sig=np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,N)))
    sig=(np.matmul(sig,sig.T)+cov*np.diag(np.abs(st.multivariate_normal.rvs(mean=0,cov=1,size=(N,1)))))
    
    diag=np.sqrt(np.diag(sig).reshape(-1,1))
    sig=(sig/np.matmul(np.abs(diag),np.abs(diag).T))
    
    num=int((N-1)*N/2)
    samp=np.random.choice(range(num),int(neg_pct*num/100))
        
    upp=np.triu_indices(N,1)
    avg_cor=np.round(np.mean(sig[upp].tolist()),2)
    max_cor=np.round(max(sig[upp].tolist()),2)
    min_cor=np.round(min(sig[upp].tolist()),2)
    upp=(upp[0][samp],upp[1][samp])
       
    sig[upp]=-sig[upp]
    sig[(upp[1],upp[0])]=-sig[(upp[1],upp[0])]
   
    return(sig,str({'neg_pct':pct_neg_cor,'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor}))

def rat_data(N):
    rat=pd.read_csv('rat.csv',sep='\t')[:,0:N]
    rat=np.cov(rat,rowvar=False)
    
    upp=np.triu_indices(N,1)
    avg_cor=np.round(np.mean(sig[upp].tolist()),2)
    max_cor=np.round(max(sig[upp].tolist()),2)
    min_cor=np.round(min(sig[upp].tolist()),2)
    pct_neg_cor=np.mean((sig[upp]>0).tolist())
    
    return(rat,{'pct_neg_cor':pct_neg_cor,'min_cor':min_cor,'avg_cor':avg_cor,'max_cor':max_cor})