import pandas as pd
import subprocess
import numpy as np
import json

from opPython.DB import *
from dataPrepPython.makePedrigreeSnps import *

def writeSnps(parms,allSnp=True):
    local=parms['local']
    etaSq=parms['parms'][0]
    numSubjects=parms['parms'][1]
    numTraits=parms['parms'][2]
    numSnps=parms['parms'][3]
    snpSeed=parms['snpSeed']
    
    snpParm=parms['snpParm']
    
    np.random.seed(snpSeed)

    np.savetxt('Y/Y.phe',np.array([['0',str(int(id_)),str(int(id_))] for id_ in range(numSubjects)]),delimiter='\t',fmt='%s')

    if 'realSnps' in snpParm:
        raw=np.round(np.loadtxt(local+'data/ail.genos.dosage.gwasSNPs.txt',delimiter='\t',dtype=str)[:,4:4+numSubjects].astype(
            float),0).astype(int)
        assert numSubjects==bimBamFmt.shape[1]
        raw=raw[np.linspace(0,len(bimBamFmt)-1,sum(numSnps)).astype(int)]
    if 'pedigreeSnps' in snpParm:
        raw=makePedigreeSnps(parms)
    if 'iidSnps' in snpParm:
        raw=np.random.choice([0,1,2],numSubjects*sum(numSnps),True,[.25,.5,.25]).reshape(sum(numSnps),-1)

    raw=raw.astype(int)
    af=np.mean(raw,axis=1)/2
    maf=np.minimum(af,1-af)
    snpData=pd.DataFrame({'chr':[snp+1 for snp,size in enumerate(numSnps) for i in range(size)],
        'ID':range(sum(numSnps)),'genetic dist': 0,'Mbp':range(sum(numSnps))})
    
    leftTab=str.maketrans('012','AAG')
    rightTab=str.maketrans('012','AGG')
    plinkFmt=np.concatenate([np.array([[0,id_,0,0,1,1] for id_ in range(numSubjects)]),
        np.char.add(np.char.add(np.char.translate(raw.T.astype(str),leftTab),' '),
        np.char.translate(raw.T.astype(str),rightTab))],axis=1)
    
    bimBamFmt=np.concatenate([np.array([[str(id_),'A','G'] for id_ in range(1,1+sum(numSnps))]),raw],axis=1)    
    
    for snp in range(1,len(numSnps)+1):
        writeChr(snpData,[snp],snp,bimBamFmt,plinkFmt,parms)

    if allSnp:
        writeChr(snpData,range(1,len(numSnps)+1),'all',bimBamFmt,plinkFmt,parms)
    
    snpData.insert(snpData.shape[1],'maf',maf)
    snpData.to_csv('snps/snpData',index=False,sep='\t')
                             
    return()

def writeChr(snpData,snpSet,name,bimBamFmt,plinkFmt,parms):
    local=parms['local']
    name=str(name)
   
    np.savetxt('snps/'+name+'.bimbam',bimBamFmt[snpData['chr'].isin(snpSet),:],delimiter='\t',fmt='%s')
    np.savetxt('snps/'+name+'.ped',plinkFmt[:,[True]*6+snpData['chr'].isin(snpSet).tolist()],delimiter='\t',fmt='%s')   
    snpData[snpData['chr'].isin(snpSet)].to_csv('snps/'+name+'.map',header=False,index=False,sep='\t')  
    np.savetxt('snps/ref-'+name,np.array([[str(id_),'A'] for id_ in range(sum(snpData['chr'].isin(snpSet)))]),
        delimiter='\t',fmt='%s')

    subprocess.call([local+'ext/plink','--file','snps/'+name,'--out','snps/'+name,'--make-bed','--noweb',
        '--reference-allele','snps/ref-'+name])
    
    return()