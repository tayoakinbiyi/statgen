
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


names=['pre_pca_hip_mouse','pre_str_hip_mouse','pre_pfc_str_mouse']
loc='../plots/pca_removed/'
dataLoc='../../data/'
N=100


# In[4]:


import sys
sys.setrecursionlimit(12000)


# In[ ]:


preds=pd.read_csv(dataLoc+'preds.csv')
preds.set_index('id')
preds=preds[['sex','batch']]


# In[ ]:


for name in names[0:1]:
    raw=pd.read_csv(dataLoc+name+'.csv')
    raw.set_index('id')
    preds_name=preds.loc[raw.index]
    raw=raw.loc[:,raw.apply(lambda x: len(np.unique(x)),axis=0)>2]
    ranks=raw.rank(axis=0,method='average')/(len(raw)+1)
    raw=ranks.apply(norm.ppf,axis=0)
        
    XtX=preds_name.T.dot(preds_name)               
    XtXinv=pd.DataFrame(np.linalg.pinv(XtX.values), index=preds_name.columns,columns=preds_name.columns)
    hat=preds_name.dot(XtXinv).dot(preds_name.T).dot(raw)
    raw=(raw-hat)
    
    for size in [0,5,10,25,50,75,95][-1:]:
        print(name,size)
        pca = PCA(n_components=100)
        Y=pca.fit_transform(raw)
        Y[:,0:size]=0
        data=pca.inverse_transform(Y)                
        
        fig,axs=plt.subplots(1,1)
        fig.set_figwidth(10)
        fig.set_figheight(5)
        axs.plot(pca.explained_variance_ratio_)
        fig.suptitle(name)
        fig.savefig(loc+str(size)+'_'+name+'_variance_explained.png',bbox_inches='tight')
        
        for method in ['average','complete','weighted']:
            # dendrogram
            print(method+'-dendrogram')
            fig,axs=plt.subplots(1,1,dpi=20)
            fig.set_figwidth(N/7,forward=True)
            fig.set_figheight(5,forward=True)
            Z=linkage(data.T, method, 'correlation')
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
            axs.imshow(np.corrcoef(data.T[den['leaves']]),interpolation='nearest', cmap='seismic',vmin=-1,vmax=1)
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




