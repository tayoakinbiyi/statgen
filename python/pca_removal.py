
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt

import scipy.stats as st
import numpy as np
import pdb
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cut_tree
from os import listdir
from scipy.stats import norm, beta

import statsmodels.formula.api as sm


# In[3]:


names=['pre_pca_str_mouse','pre_pca_pfc_mouse'] #'pre_pca_hip_mouse',
loc='../plots/pca_removed/'
dataLoc='../../data/'


# In[4]:


import sys
sys.setrecursionlimit(12000)


# In[ ]:


preds=pd.read_csv(dataLoc+'preds.csv')
preds.set_index('id')
preds=preds[['sex','batch']]
preds.insert(0,'intercept',1)

# In[ ]:


for name in names:
    raw=pd.read_csv(dataLoc+name+'.csv')
    N=raw.shape[1]
    raw.set_index('id')
    preds_name=preds.loc[raw.index]
    raw=raw.loc[:,raw.apply(lambda x: len(np.unique(x)),axis=0)>2]
    ranks=raw.rank(axis=0,method='average')/(len(raw)+1)
    raw=ranks.apply(norm.ppf,axis=0)
        
    XtX=preds_name.T.dot(preds_name)               
    XtXinv=pd.DataFrame(np.linalg.pinv(XtX.values), index=preds_name.columns,columns=preds_name.columns)
    hat=preds_name.dot(XtXinv).dot(preds_name.T).dot(raw)
    raw=(raw-hat)
    
    U,D,Vt=np.linalg.svd(raw.values)
    
    for size in [0,5,10,25,50,75,95]:
        print(name,size)
        if size>0:
            X=pd.DataFrame(U[:,0:size],columns=range(size),index=raw.index)
            XtX=X.T.dot(X)
            XtXinv=pd.DataFrame(np.linalg.pinv(XtX.values), index=X.columns,columns=X.columns)
            hat=X.dot(XtXinv).dot(X.T).dot(raw)
            data=raw-hat
        else:
            data=raw
        
        for method in ['average','complete','weighted']:
            # dendrogram
            print(method+'-dendrogram')
            fig,axs=plt.subplots(1,1,dpi=20)
            fig.set_figwidth(N/7,forward=True)
            fig.set_figheight(5,forward=True)
            Z=linkage(data.T.values, method, 'correlation')
            den=dendrogram(Z, color_threshold=0,ax=axs)
            axs.tick_params(axis='X',labelsize=20)
            fig.suptitle(name+'_'+method)
            fig.savefig(loc+str(size)+'_'+name+'_'+method+'_dendogram.png',bbox_inches='tight')
            plt.close()

            # corr plot
            print(method+'-corr')
            fig,axs=plt.subplots(1,1)
            fig.set_figwidth(30,forward=True)
            fig.set_figheight(30,forward=True)
            axs.imshow(np.corrcoef(data.T.values[den['leaves']]),interpolation='nearest', cmap='seismic',vmin=-1,vmax=1)
            fig.suptitle(name+'_'+method)
            fig.savefig(loc+str(size)+'_'+name+'_'+method+'_corr.png',bbox_inches='tight')
            plt.close()    

        fig,axs=plt.subplots(1,1)
        fig.set_figwidth(7,forward=True)
        fig.set_figheight(7,forward=True)
        off_diag=np.corrcoef(data,rowvar=False)[np.triu_indices(N,1)].flatten()  
        axs.hist(off_diag,bins=np.linspace(-1,1,100))
        fig.suptitle(name)
        fig.savefig(loc+str(size)+'_'+name+'_full_off_diag.png',bbox_inches='tight')
        plt.close()    
    
    


# In[ ]:





# In[ ]:





# In[ ]:




