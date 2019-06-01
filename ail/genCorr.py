import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

from statsmodels.stats.moment_helpers import cov2corr
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from collections import Counter
from scipy.stats import norm

import numpy as np
import pandas as pd
import pdb
import os  

home='/home/ubuntu/ail/'

files={
    'dataDir':home+'data/',
    'scratchDir':home+'scratch/',
    'gemma':home+'gemma'
}
numPCs=10


#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

def corrHelp(trait1,trait2,ind,col):
    if os.path.isfile('cov-'+trait1+'-'+trait2+'.csv'):
        return()
    
    chrs=['chr'+str(x) for x in range(1,20) if (trait1!='chr'+str(x)) and (trait2!='chr'+str(x))]
    
    N=0
    mean1=np.array([0]*len(ind))
    mean2=np.array([0]*len(col))
    
    for snpChr in chrs:
        df1=pd.read_csv('pvals-final-'+snpChr+'-'+trait1+'.txt',index_col=[0,1],header=[0,1,2]).values 
        df2=pd.read_csv('pvals-final-'+snpChr+'-'+trait2+'.txt',index_col=[0,1],header=[0,1,2]).values

        df1=-norm.ppf(df1/2)
        df2=-norm.ppf(df2/2)

        mean1+=np.sum(df1,axis=0)
        mean2+=np.sum(df2,axis=0)
    
        N+=df1.shape[0]
        
    sum1/=N
    sum2/=N
    
    for snpChr in chrs:        
        df1=pd.read_csv('pvals-final-'+snpChr+'-'+trait1+'.txt',index_col=[0,1],header=[0,1,2]).values 
        df2=pd.read_csv('pvals-final-'+snpChr+'-'+trait2+'.txt',index_col=[0,1],header=[0,1,2]).values

        df1=-norm.ppf(df1/2)
        df2=-norm.ppf(df2/2)
        
        df1=df1-mean1
        df2=df2-mean2
    
        if N==0:
            cov=pd.DataFrame(np.matmul(df1.T,df2),index=ind,columns=col)
        else:
            cov+=pd.DataFrame(np.matmul(df1.T,df2),index=ind,columns=col)
                            
    cov/=(N-1)
    cov.to_csv('cov-'+trait1+'-'+trait2+'.csv')
    
    print(trait1,trait2,'done')
    
if __name__ == '__main__':    
    M=20

    expr=pd.read_csv(dataDir+'hipNormCounts.expr.txt',sep='\t',index_col=0,header=0)
    traitInfo=pd.read_csv(dataDir+'traitInfo.csv',header=0,index_col=None)
    tab=pd.DataFrame(Counter(traitInfo.trait.values),index=[0]).T
    tab=tab[tab[0]>1]
    dups=tab.index.values.flatten()
    traitInfo=traitInfo[~traitInfo.trait.isin(dups)]
    traitChr=pd.DataFrame({'trait':expr.columns}).merge(traitInfo,on='trait')
    traitChr=traitChr[traitChr.chromosome.isin(['chr'+str(x) for x in range(1,M)])]

    genCorr=False
    genFinal=False
    makeCorr=True
    
    pvals=pd.DataFrame()
    os.chdir(scratchDir)
    
    if genCorr:
        futures=[]
        with ProcessPoolExecutor() as executor: 
            for i in range(1,M):
                trait1='chr'+str(i)
                ind=pd.MultiIndex.from_tuples(traitChr[traitChr.chromosome==trait1].values.tolist(),names=['trait','chr','Mbp'])
                for j in range(i,M):
                    trait2='chr'+str(j)
                    col=pd.MultiIndex.from_tuples(traitChr[traitChr.chromosome==trait2].values.tolist(),names=['trait','chr','Mbp'])
                    
                    futures.append(executor.submit(corrHelp,trait1,trait2,ind,col))
        
        wait(futures,return_when=ALL_COMPLETED)

    if genFinal:

        cov=pd.DataFrame(index=pd.MultiIndex.from_tuples(traitChr.values.tolist(),names=['trait','chr','Mbp']),
                          columns=pd.MultiIndex.from_tuples(traitChr.values.tolist(),names=['trait','chr','Mbp']))      
        
        for i in range(1,M):
            for j in range(i,M):
                print(i,j)
                cov.loc[(slice(None),'chr'+str(i),slice(None)),(slice(None),'chr'+str(j),slice(None))]=pd.read_csv(
                    'cov-chr'+str(i)+'-chr'+str(j)+'.csv',index_col=[0,1,2],header=[0,1,2]).values
                if i!=j:
                    cov.loc[(slice(None),'chr'+str(j),slice(None)),(slice(None),'chr'+str(i),slice(None))]=pd.read_csv(
                        'cov-chr'+str(i)+'-chr'+str(j)+'.csv',index_col=[0,1,2],header=[0,1,2]).T.values
        
        cov.to_csv('cov.csv')
        
    if makeCorr:
        cov=pd.read_csv('cov.csv',index_col=[0,1,2],header=[0,1,2]).values
        U,D,Vt=np.linalg.svd(cov)

        bias=min(D)
        if bias <0:
            print('bias ='+str(bias))
            D-=bias

            cov=np.matmul(np.matmul(U,np.diag(D)),U.T)

        corr=cov2corr(cov)

        fig,ax=plt.subplots(1,2)
        N=len(cov)        
        
        ax[0].hist(corr[np.triu_indices(N,1)])
        ax[0].set_xlim([-1,1])
        ax[0].set_title('Off Diagonal')
        ax[1].hist(np.diag(cov))
        ax[1].set_title('Cov Diagonal')
        
        fig.savefig('../corr_plot.png')
        
        np.savetxt('corr.csv',corr,delimiter=',')
    
            