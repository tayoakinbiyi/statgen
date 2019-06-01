import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from multiprocessing import cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from python.lmm import *
from python.grm import *

#home='/phddata/akinbiyi/'
home='/project/abney/'

predExpr=True
genPC=True
GRM=True
doLMM=True

files={
    'dataDir':home+'ail/data/',
    'scratchDir':home+'ail/scratch/',
    'gemma':home+'ail/gemma'
}
numPCs=100


#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

# snps.txt
print('load raw')
snps=pd.read_csv(dataDir+'ail.genos.dosage.gwasSNPs.txt',sep='\t',header=None,index_col=None)
preds=pd.read_csv(dataDir+'ail.phenos.final.txt',sep='\t',header=0,index_col=0)
expr=pd.read_csv(dataDir+'hipNormCounts.expr.txt',sep='\t',index_col=0,header=0)

print('load traitInfo')
traitInfo=pd.read_csv(dataDir+'traitInfo.csv',header=0,index_col=None)
tab=pd.DataFrame(Counter(traitInfo.trait.values),index=[0]).T
tab=tab[tab[0]>1]
dups=tab.index.values.flatten()
traitInfo=traitInfo[~traitInfo.trait.isin(dups)]

print('get ids')
# get chromosome of each trait
traitChr=pd.DataFrame({'trait':expr.columns}).merge(traitInfo,on='trait')
expr=expr[traitChr.trait]

# get list of person IDs
allIds=preds.index.values.flatten().tolist()
exprIds=expr.index.values.flatten().tolist()

# format SNP file
snps.columns=['chr','snpId','minor','major']+allIds
snpChr=snps['chr'].values.flatten()
snpId=snps['snpId'].values.flatten()
snps.drop(columns='chr',inplace=True)
snps=snps[['snpId','minor','major']+exprIds]

if predExpr:
    # recover sex and batch data
    print('prep covs')
    preds =preds.loc[exprIds,['sex','batch']]
    preds.insert(0,'intercept',1)
    preds=preds.values

    # quantile normalize expr data
    print('quant normalize')
    ranks=expr.rank(axis=0,method='average')/(len(expr)+1)
    expr=ranks.apply(norm.ppf,axis=0).values

    # remove sex and batch
    print('remove batch, sex')
    reg=MultiOutputRegressor(LinearRegression(),n_jobs=-1).fit(preds,expr)
    expr=(expr-reg.predict(preds))

    print('write traits to file')
    for trait in set(traitChr.chromosome):
        np.savetxt(scratchDir+'pheno-'+trait+'.txt',expr[:,traitChr.chromosome==trait],delimiter='\t')
        
    # generate PCs
    print('generate PCs')
    U,D,Vt=np.linalg.svd(expr)
    PCs=np.concatenate([np.ones([len(expr),1]),U[:,0:numPCs]],axis=1)

    # write covariate data to file
    print('write PCs to file')
    np.savetxt(scratchDir+'PCAll.txt',PCs,delimiter='\t')

if GRM:
    np.savetxt(scratchDir+'dummy.txt',np.ones([len(expr),1]),delimiter='\t')

    print('generate grm and genome files')
    os.chdir(scratchDir)

    futures=[]
    with ProcessPoolExecutor() as executor: 
        for ch in set(snpChr):
            futures.append(executor.submit(grm,ch,snpChr,snps,files))

    wait(futures,return_when=ALL_COMPLETED)
    
if genPC:
    print('Find PCs')
    os.chdir(scratchDir)
    
    futures=[]
    with ProcessPoolExecutor() as executor: 
        for snp in set(snpChr):
            futures.append(executor.submit(genPCPval,snp,numPCs,files))

    wait(futures,return_when=ALL_COMPLETED)
    print('finished genPCPval')
    
    PCPval=[]
    count=0
    for snp in set(snpChr):
        PCPval[count]=np.loadtxt('pvals-PC-'+snp+'.txt',delimiter='\t')
        
    PCPval=np.concatenate(PCPval,axis=0)
        
    whichPCs=np.arange(numPCs)[np.min(PCPval,axis=0)>9.01e-6]
    print('PCs ',whichPCs)
    
    PCs=np.savetxt(scratchDir+'PC.txt',np.loadtxt('PCAll.txt',delimiter='\t')[:,whichPCs],delimiter='\t')

# loop through traits and chromosomes
if doLMM:
    print('future loop')
    os.chdir(scratchDir)
    
    futures=[]
    with ProcessPoolExecutor() as executor: 
        for snp in set(snpChr):
            for trait in set(traitChr.chromosome):
                futures.append(executor.submit(lmm,snp,trait,snpId[snpChr==snp],traitChr.loc[traitChr.chromosome==trait],files))

    wait(futures,return_when=ALL_COMPLETED)

