import pandas as pd
import numpy as np
import subprocess

from ail.opPython.DB import *

def genSnps(parms):
    local=parms['local']
    name=parms['name']
    simSnpSize=parms['simSnpSize']
    
    pd.DataFrame({'# name':np.arange(50000),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
        local+name+'geneDrop/map.txt',sep='\t',index=False)

    mouseIds=DBRead(name+'process/mouseIds',parms,True)
    
    pd.DataFrame({'id':[str(Id)+'.1' for Id in mouseIds]}).to_csv(local+name+'geneDrop/sampleIds.txt',index=False,header=False)
    
    DBUpload(name+'geneDrop/map.txt',parms,False)
    DBUpload(name+'geneDrop/sampleIds.txt',parms,False)
    
    path=local+name

    pd.DataFrame({'parms':['./ext/ail_revised.ped.txt',path+'geneDrop/sampleIds.txt',path+'geneDrop/map.txt',0,0]}).to_csv(path+
        'geneDrop/parms.txt',index=False,header=None)

    cmd=['./ext/gdrop','-p',path+'geneDrop/parms.txt','-s','1','-o',path+'geneDrop/geneDrop.txt']
    subprocess.run(cmd)
    print('finished gdrop',flush=True)
    
    geneDrop=pd.read_csv(path+'geneDrop/geneDrop.txt',header=0,sep='\t').set_index([0])    
    snps=pd.merge(pd.DataFrame({'id':range(len(sim)),'major':'A','minor':'G'},index=[0]),geneDrop,left_index=True,right_index=True)
        
    snps.to_csv(local+name+'geneDrop/geno-1.txt',sep='\t',index=False,header=False)        
    DBUpload(name+'geneDrop/geno-1.txt',parms,toPickle=False)

    return()