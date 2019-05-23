import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from python.lmm import *

files={
    'dataDir':'/phddata/akinbiyi/ail/data/',
    'scratchDir':'/phddata/akinbiyi/ail/scratch/',
    'gemma':'/phddata/akinbiyi/ail/gemma'
}
numPCs=10


#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

# snps.txt
print('load raw')
snps=pd.read_csv(dataDir+'ail.genos.dosage.gwasSNPs.txt',sep='\t',header=None,index_col=None)
preds=pd.read_csv(dataDir+'ail.phenos.final.txt',sep='\t',header=0,index_col=0)
expr=pd.read_csv(dataDir+'hipNormCounts.expr.txt',sep='\t',index_col=0,header=0)

print('traitInfo')
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

# recover sex and batch data
print('prep covs')
preds =preds.loc[exprIds,['sex','batch']]
preds.insert(0,'intercept',1)

# quantile normalize expr data
print('quant normalize')
ranks=expr.rank(axis=0,method='average')/(len(expr)+1)
expr=ranks.apply(norm.ppf,axis=0)

# remove sex and batch
print('remove batch, sex')
XtX=preds.T.dot(preds)               
XtXinv=pd.DataFrame(np.linalg.pinv(XtX.values), index=preds.columns,columns=preds.columns)
hat=preds.dot(XtXinv).dot(preds.T).dot(expr)
expr=(expr-hat)
 
# generate PCs
print('generate PCs')
U,D,Vt=np.linalg.svd(expr.values)
PCs=pd.DataFrame(U[:,0:numPCs],columns=range(numPCs),index=expr.index)
PCs.insert(0,'intercept',1)

# set the current directory
os.chdir(scratchDir)

# write covariate data to file
print('write PCs to file')
PCs.to_csv('PC.txt',header=False,index=False,sep='\t')
expr.to_csv('pheno.txt',sep='\t',index=False,header=False)

# create dataframe to hold all pvals
traitChr=traitChr.iloc[0:5,:]
allRes=pd.DataFrame(columns=pd.MultiIndex.from_tuples(traitChr.values.tolist(),names=['trait','chr','Mbp']))

# oneChrFunc('chr1',snpChr,snpId,snps,traitChr,files)
# pdb.set_trace()

for ch in set(snpChr):
    snps[snpChr!=ch].to_csv('geno.txt',sep=' ',index=False,header=False)
    
    # generate loco
    subprocess.run([files['gemma'],'-g','geno.txt','-p','pheno.txt','-gk','2','-o','grm-'+str(ch)])

    # move grm to scratch
    os.rename('output/grm-'+str(ch)+'.sXX.txt','grm-'+str(ch)+'.sXX.txt')

    # write chr gene file
    snps[snpChr==ch].to_csv('geno-'+str(ch)+'.txt',sep=' ',index=False,header=False)
    
# loop through traits and chromosomes
print('future loop')
futures=[]
with ProcessPoolExecutor() as executor: 
    for ch in set(snpChr):
        futures.append(executor.submit(lmm,ch,snpChr,snpId,traitChr,files))

for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
    result=f.result()
    allRes=allRes.append(result)
    
allRes.index=pd.MultiIndex.from_tuples(allRes.index,names=['chr','Mbp'])
allRes.to_csv(dataDir+'allRes.csv')

        
