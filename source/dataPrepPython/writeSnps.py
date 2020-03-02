import pandas as pd
import subprocess
import numpy as np
import json

from opPython.DB import *

def writeInputs(bimBamFmt,Y,parms):
    snpChr=parms['snpChr']
    local=parms['local']
    fileFmt=parms['fileFmt']
    snpSize=parms['snpSize']
    
    assert fileFmt in ['bimbam','plink']
    
    mouseIds=np.arange(len(pedSnps))
    af=np.mean(bimBamFmt,axis=1)/2
    maf=np.min(af,1-af)
    snpData=pd.DataFrame({'chr':[snp for snp,size in zip(snpChr,snpSize) for ind in range(size)],
                          'ID':range(numSnps),'genetic dist': 0,'Mbp':range(numSnps)})
    
    leftTab=str.maketrans('012','AAG')
    rightTab=str.maketrans('012','AGG')
    plinkFmt=np.char.add(np.char.add(np.char.translate(bimBamFmt[snpData['chr'].values==snp,:].T,leftTab),' '),
                         np.char.translate(bimBamFmt[snpData['chr'].values==snp,:].T,rightTab))
       
    for snp in snpChr:
        writeChr(snpData,snp,snp,bimBamFmt,plinkFmt)

    writeChr(snpData,snpChr,'all',bimBamFmt,plinkFmt)

    snpData.insert(snpData.shape[1],'maf',maf)
    snpData.to_csv('ped/snpData',index=False,sep='\t')

    np.savetxt('input/Y.txt',np.array([[0,id_,0,0,1,row] for id_,row in enumerate(Y)]),delimiter='\t')

    YCorr=np.corrcoef(Y,rowvar=False)
    np.savetxt('LZCorr/LTraitCorr',makePSD(YCorr),delimiter='\t')

    return()

def writeChr(snpData,snp,name,bimBamFmt,plinkFmt):
    snp=str(snp)
    
    fam=np.array([[0,id_,0,0,1,1] for id_ in range(len(plinkFmt))])
    np.savetxt('ped/snp-'+name+'.bimbam',np.concatenate([np.array([[id_,'A','G'] for id_ in range(sum(snpData['chr'].isin(snp)))]),
        bimBamFmt],axis=1),delimiter='\t')
    np.savetxt('ped/snp-'+name+'.ped',np.concatenate([fam,plinkFmt],axis=1).to_csv('ped/snp-'+name+'.ped',header=False,
        index=False,sep='\t')   
    snpData[snpData['chr'].isin(snp)].to_csv('ped/snp-'+name+'.map',header=False,index=False,sep='\t')  
    np.savetxt('ped/ref-'+name,np.array([[id_,'A'] for id_ in range(sum(snpData['chr'].isin(snp)))]),delimiter='\t')
    subprocess.call([local+'ext/plink','--file','ped/snp-'+name,'--out','ped/snp-'+name,'--make-bed','--noweb',
        '--reference-allele','ped/ref-'+name])
    
    return()