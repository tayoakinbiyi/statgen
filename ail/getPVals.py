import pandas as pd
import numpy as np
import subprocess
import os
import pdb
from scipy.stats import norm
from collections import Counter

dataDir='/phddata/akinbiyi/ail/data/'
scratchDir='/phddata/akinbiyi/ail/scratch/'
gemma='/phddata/akinbiyi/ail/gemma'
numPCs=10

#load raw files
#ail.genos.dosage.gwasSNPs.txt
snps=pd.read_csv(dataDir+'snps.txt',sep='\t',header=None,index_col=None)
preds=pd.read_csv(dataDir+'ail.phenos.final.txt',sep='\t',header=0,index_col=0)
expr=pd.read_csv(dataDir+'hipNormCounts.expr.txt',sep='\t',index_col=0,header=0)

traitInfo=pd.read_csv(dataDir+'traitInfo.csv',header=0,index_col=None)
tab=pd.DataFrame(Counter(traitInfo.trait.values),index=[0]).T
tab=tab[tab[0]>1]
dups=tab.index.values.flatten()
traitInfo=traitInfo[~traitInfo.trait.isin(dups)]

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
preds =preds.loc[exprIds,['sex','batch']]
preds.insert(0,'intercept',1)

# quantile normalize expr data
ranks=expr.rank(axis=0,method='average')/(len(expr)+1)
expr=ranks.apply(norm.ppf,axis=0)

# remove sex and batch
XtX=preds.T.dot(preds)               
XtXinv=pd.DataFrame(np.linalg.pinv(XtX.values), index=preds.columns,columns=preds.columns)
hat=preds.dot(XtXinv).dot(preds.T).dot(expr)
expr=(expr-hat)
 
# generate PCs
U,D,Vt=np.linalg.svd(expr.values)
PCs=pd.DataFrame(U[:,0:numPCs],columns=range(numPCs),index=expr.index)
PCs.insert(0,'intercept',1)

'''
bim bam:
    mean genotype file (snps)
        rs1, A, T, 0.02, 0.80, 1.50
        rs2, G, C, 0.98, 0.04, 1.00
        
    phenotype file (expr)
        1.2 -0.3 -1.5
        NA 1.5 0.3
        2.7 1.1 NA
        -0.2 -0.7 0.8
        3.3 2.4 2.1
'''

# set the current directory
os.chdir(scratchDir)

# write covariate data to file

PCs.to_csv('PC.txt',header=False,index=False,sep='\t')

# create dataframe to hold all pvals

genoArray=[snpChr,snpId]
genoTuple = list(zip(*genoArray))
allRes=pd.DataFrame(columns=pd.MultiIndex.from_tuples(traitChr.values.tolist(),names=['trait','chr','Mbp']),
                    index=pd.MultiIndex.from_tuples(genoTuple, names=['chr', 'Mbp']))

expr.to_csv('pheno.txt',sep='\t',index=False,header=False)

# loop through traits and chromosomes
futures=[]
with ProcessPoolExecutor() as executor: 
    for k in range(d):
        futures.append(executor.submit(ggHelp,z[k],ggnullDat[k],k))

ggnull=pd.DataFrame(dtype='float32')
for f in wait(futures,return_when=FIRST_COMPLETED)[0]:
    result=f.result()
    ggnull.insert(ggnull.shape[1],result[0],result[1])
for ch in set(snpChr):
    # generate loco
    snps[snpChr!=ch].to_csv('geno.txt',sep=' ',index=False,header=False)
    subprocess.run([gemma,'-g','geno.txt','-p','pheno.txt','-gk','2','-o','grm'])

    # move grm to scratch
    os.rename(scratchDir+'output/grm.sXX.txt',scratchDir+'grm.sXX.txt')

    # run 
    snps[snpChr==ch].to_csv('geno.txt',sep=' ',index=False,header=False)

    for ph in range(expr.shape[1]):
        if ch==traitInfo.chromosome.iloc[ph]:
            continue
            
        print(ph,ch)

        subprocess.run([gemma,'-g','geno.txt','-p','pheno.txt','-c','PC.txt','-lmm','3','-o','pvals','-k','grm.sXX.txt',
                        '-maf','0.05','-r2','0.99','-n',str(ph+1)])
        
        #recover p-vals
        pvals=pd.read_csv('output/pvals.assoc.txt',sep='\t')
        
        # extract p_vals        
        allRes.loc[ch,traitInfo.trait.iloc[ph]]=pvals['p_score'].values
        

allRes.to_csv(dataDir+'allRes.csv')