import pandas as pd
import numpy as np
import subprocess
import random

from ail.opPython.DB import *

def genSnps(parms):
    local=parms['local']
    name=parms['name']
    H0SnpSize=parms['H0SnpSize']
    H1SnpSize=parms['H1SnpSize']
    
    path=local+name
    
    mouseIds=DBRead(name+'process/mouseIds',parms,True)
    
    pd.DataFrame({'id':[str(Id)+'.1' for Id in mouseIds]}).to_csv(local+name+'geneDrop/sampleIds.txt',index=False,header=False)
    
    DBUpload(name+'geneDrop/sampleIds.txt',parms,False)
    
    pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt',path+'geneDrop/sampleIds.txt',path+'geneDrop/map.txt',0,0]}).to_csv(path+
        'geneDrop/parms.txt',index=False,header=None)
    
    snpData=pd.DataFrame({'chr':['chr0']*H0SnpSize+['chr1']*H1SnpSize,'Mbp':range(H0SnpSize+H1SnpSize)})
    DBWrite(snpData,name+'process/snpData',parms,True)
            
    getSnpsHelp('chr0',H0SnpSize,parms)
    getSnpsHelp('chr1',H1SnpSize,parms)

    return()

def getSnpsHelp(snp,size,parms):
    local=parms['local']
    name=parms['name']

    path=local+name

    snps=[]
    numSnps=0
    while numSnps<size:
        newAdd=min(10000,size-numSnps)
        pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
            local+name+'geneDrop/map.txt',sep='\t',index=False)
        DBUpload(name+'geneDrop/map.txt',parms,False)

        cmd=['./ext/gdrop','-p',path+'geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o',path+'geneDrop/geneDrop']
        subprocess.run(cmd)
    
        val=pd.read_csv(path+'geneDrop/geneDrop.geno_true',header=0,sep='\t').iloc[:,4:]
        val=np.concatenate([(col.str.split(' ',expand=True).astype(int)>2).sum(axis=1).values.reshape(-1,1) for ind,
            col in val.iteritems()],axis=1)
        
        maf=np.mean(val,axis=1)/2
        maf=np.minimum(maf,1-maf)
        val=val[maf>.1,:]
                
        numSnps+=val.shape[0]
        print('removed '+str(newAdd-val.shape[0])+' snps',flush=True)

        snps+=[val]
    
    snps=pd.DataFrame(np.concatenate(snps,axis=0)).set_index(np.array([0]*size))
    snps=pd.merge(pd.DataFrame({'major':'A','minor':'G'},index=[0]),snps,left_index=True,right_index=True)
    snps.insert(0,'id',range(len(snps)))
        
    snps.to_csv(local+name+'process/geno-'+snp+'.txt',sep='\t',index=False,header=False)        
    DBUpload(name+'process/geno-'+snp+'.txt',parms,toPickle=False)

    return()