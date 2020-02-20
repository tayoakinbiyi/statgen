import pandas as pd
import subprocess
import numpy as np
import json

from opPython.DB import *

def writeInputs(bimBamFmt,Y,parms):
    snpChr=parms['snpChr']
    local=parms['local']
    snpSize=parms['snpSize']
    numSnps=len(bimBamFmt)
        
    mouseIds=np.arange(len(bimBamFmt))
    af=np.mean(bimBamFmt,axis=1)/2
    maf=np.minimum(af,1-af)
    snpData=pd.DataFrame({'chr':[snp for snp,size in zip(snpChr,snpSize) for ind in range(size)],
        'ID':range(numSnps),'genetic dist': 0,'Mbp':range(numSnps)})
    
    leftTab=str.maketrans('012','AAG')
    rightTab=str.maketrans('012','AGG')
    plinkFmt=np.char.add(np.char.add(np.char.translate(bimBamFmt.T.astype(str),leftTab),' '),
        np.char.translate(bimBamFmt.T.astype(str),rightTab))
    
    for snp in snpChr:
        writeChr(snpData,[snp],snp,bimBamFmt,plinkFmt,parms)

    writeChr(snpData,snpChr,'all',bimBamFmt,plinkFmt,parms)
    
    snpData.insert(snpData.shape[1],'maf',maf)
    snpData.to_csv('inputs/snpData',index=False,sep='\t')
    
    np.savetxt('inputs/Y.phe',np.array([['0',str(int(id_))]+row for id_,row in enumerate(Y.tolist())]),delimiter='\t',fmt='%s')
                         
    return()

def writeChr(snpData,snp,name,bimBamFmt,plinkFmt,parms):
    local=parms['local']
    name=str(name)
    
    fam=np.array([[0,id_,0,0,1,1] for id_ in range(len(plinkFmt))])
    np.savetxt('inputs/'+name+'.bimbam',np.array([[str(id_),'A','G']+row for id_,row in enumerate(bimBamFmt[snpData['chr'].isin(snp)
        ].tolist())]),delimiter='\t',fmt='%s')
    np.savetxt('inputs/'+name+'.ped',np.concatenate([fam,plinkFmt],axis=1),delimiter='\t',fmt='%s')   
    snpData[snpData['chr'].isin(snp)].to_csv('inputs/'+name+'.map',header=False,index=False,sep='\t')  
    np.savetxt('inputs/ref-'+name,np.array([[str(id_),'A'] for id_ in range(sum(snpData['chr'].isin(snp)))]),delimiter='\t',fmt='%s')
    subprocess.call([local+'ext/plink','--file','inputs/'+name,'--out','inputs/'+name,'--make-bed','--noweb',
        '--reference-allele','inputs/ref-'+name])
    
    return()