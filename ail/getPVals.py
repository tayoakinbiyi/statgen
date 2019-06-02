import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from multiprocessing import cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from collections import Counter

from python.lmm import *
from python.grm import *
from python.genPCPval import *

#home='/phddata/akinbiyi/'
home='/project/abney/'

responseFile='hipRaw'
predWrite=False
genoWrite=False
removePCs=True
GRM=False
writeExpr=False
doLMM=False
numPCs=100

#########################################################################################33

files={
    'dataDir':home+'ail/data/',
    'scratchDir':home+'ail/scratch/',
    'gemma':home+'ail/gemma'
}

#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

print('load raw')
expr=pd.read_csv(dataDir+responseFile+'.txt',sep='\t',index_col=0,header=0)

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
exprIds=expr.index.values.flatten().tolist()

os.chdir(scratchDir)

if predWrite:
    preds=pd.read_csv(dataDir+'ail.phenos.final.txt',sep='\t',header=0,index_col=0)

    allIds=preds.index.values.flatten().tolist()
    np.savetxt('allId.csv',allIds,delimiter=',')

    # recover sex and batch data
    print('prep covs')
    preds =preds.loc[exprIds,['sex','batch']]
    preds.insert(0,'intercept',1)
    
    preds=preds.values
    np.savetxt('preds.txt',preds,delimiter='\t')
    
else:
    allIds=np.loadtxt('allId.csv',delimiter=',')

if genoWrite:
    snps=pd.read_csv(dataDir+'ail.genos.dosage.gwasSNPs.txt',sep='\t',header=None,index_col=None)
    
    # format SNP file
    snps.columns=['chr','snpId','minor','major']+allIds
    snpChr=snps['chr'].values.flatten()
    snpId=snps['snpId'].values.flatten()
    snps.drop(columns='chr',inplace=True)
    snps=snps[['snpId','minor','major']+exprIds]

    print('write one snp geno files')
    for snp in set(snpChr):
        snps[snpChr==snp].to_csv('geno-'+snp+'.txt',sep='\t',index=False,header=False)
    
    pd.DataFrame({'id':snpId,'chr':snpChr}).to_csv(dataDir+'snpId.csv',index=False)
else:
    snpData=pd.read_csv(dataDir+'snpId.csv')
    snpId=snpData['id']
    snpChr=snpData['chr']
    
if GRM:
    np.savetxt('dummy.txt',np.ones([len(expr),1]),delimiter='\t')

    print('generate grm and genome files')

    futures=[]
    with ProcessPoolExecutor() as executor: 
        for snp in set(snpChr):
            futures.append(executor.submit(grm,snp,snpChr,snps,files))

    wait(futures,return_when=ALL_COMPLETED)
    
if removePCs:
    # quantile normalize expr data
    print('quant normalize')
    ranks=expr.rank(axis=0,method='average')/(len(expr)+1)
    expr=ranks.apply(norm.ppf,axis=0).values

    # generate PCs
    print('generate PCs')
    U,D,Vt=np.linalg.svd(expr)
    PCs=U[:,0:numPCs]

    # write covariate data to file
    print('write PCs to file')
    np.savetxt('PCAll.txt',PCs,delimiter='\t')

    print('Find PCs')
    futures=[]
    with ProcessPoolExecutor() as executor: 
        for snp in set(snpChr):
            futures.append(executor.submit(genPCPval,snp,snpId[snpChr==snp],numPCs,files))
    
    wait(futures,return_when=ALL_COMPLETED)
    print('finished genPCPval')
    
    PCPval=[]
    for snp in set(snpChr):
        PCPval+=[np.loadtxt('pvals-PC-'+snp+'.txt',delimiter='\t')]
        
    PCPval=np.concatenate(PCPval,axis=0)
    pdb.set_trace()
    whichPCs=np.arange(numPCs)[np.min(PCPval,axis=0)>9.01e-6]
    print('PCs ',whichPCs,' # ',len(whicPCs))
        
    # remove PCs
    print('PCs')
    reg=MultiOutputRegressor(LinearRegression(),n_jobs=-1).fit(PCs[:,whichPCs],expr)
    expr=(expr-reg.predict(PCs[:,whichPCs]))    

    print('quant normalize')
    ranks=expr.rank(axis=0,method='average')/(len(expr)+1)
    expr=ranks.apply(norm.ppf,axis=0).values
    
    var=np.diag(np.cov(expr,rowvar=True))

if writeExpr:
    print('write traits to file')
    for trait in set(traitChr.chromosome):
        np.savetxt('pheno-'+trait+'.txt',expr[:,traitChr.chromosome==trait],delimiter='\t')        

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

