import pandas as pd
import subprocess
import numpy as np
import json

from opPython.DB import *

def writeInputs(bimBamFmt,parms):
    local=parms['local']
    numSnps=parms['parms'][-1]
    numSubjects=parms['parms'][-3]
        
    af=np.mean(bimBamFmt,axis=1)/2
    maf=np.minimum(af,1-af)
    snpData=pd.DataFrame({'chr':[snp+1 for snp,size in enumerate(numSnps) for i in range(size)],
        'ID':range(sum(numSnps)),'genetic dist': 0,'Mbp':range(sum(numSnps))})
    
    leftTab=str.maketrans('012','AAG')
    rightTab=str.maketrans('012','AGG')
    plinkFmt=np.concatenate([np.array([[0,id_,0,0,1,1] for id_ in range(numSubjects)]),
        np.char.add(np.char.add(np.char.translate(bimBamFmt.T.astype(str),leftTab),' '),
        np.char.translate(bimBamFmt.T.astype(str),rightTab))],axis=1)
    
    bimBamFmt=np.concatenate([np.array([[str(id_),'A','G'] for id_ in range(1,1+sum(numSnps))]),bimBamFmt],axis=1)    
    
    for snp in range(1,len(numSnps)+1):
        writeChr(snpData,[snp],snp,bimBamFmt,plinkFmt,parms)

    writeChr(snpData,range(1,len(numSnps)+1),'all',bimBamFmt,plinkFmt,parms)
    
    snpData.insert(snpData.shape[1],'maf',maf)
    snpData.to_csv('inputs/snpData',index=False,sep='\t')
                             
    return()

def writeChr(snpData,snpSet,name,bimBamFmt,plinkFmt,parms):
    local=parms['local']
    name=str(name)
    
    np.savetxt('inputs/'+name+'.bimbam',bimBamFmt[snpData['chr'].isin(snpSet),:],delimiter='\t',fmt='%s')
    np.savetxt('inputs/'+name+'.ped',plinkFmt[:,[True]*6+snpData['chr'].isin(snpSet).tolist()],delimiter='\t',fmt='%s')   
    snpData[snpData['chr'].isin(snpSet)].to_csv('inputs/'+name+'.map',header=False,index=False,sep='\t')  
    np.savetxt('inputs/ref-'+name,np.array([[str(id_),'A'] for id_ in range(sum(snpData['chr'].isin(snpSet)))]),
        delimiter='\t',fmt='%s')

    subprocess.call([local+'ext/plink','--file','inputs/'+name,'--out','inputs/'+name,'--make-bed','--noweb',
        '--reference-allele','inputs/ref-'+name])
    
    return()