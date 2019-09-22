import pandas as pd
import numpy as np
from ail.opPython.DB import *
import subprocess

def genGRM(snp,snps,parms):
    local=parms['local']
    name=parms['name']
    grmParm=parms['grmParm']
    
    if DBIsFile(name+'process','grm-'+snp+'.txt',parms):
        return()
    
    snps.to_csv(local+name+'process/geno-grm-'+snp+'.txt',index=None,header=None,sep='\t')
    
    # generate loco
    print('grm',snp,flush=True)
    cmd=['./ext/gemma','-g',local+name+'process/geno-grm-'+snp+'.txt','-gk',('1' if grmParm=='c' else '2'),
        '-o',name[:-1]+'-grm-'+snp,'-p',local+name+'process/dummy.txt']
    subprocess.run(cmd)

    # move grm to scratch
    os.remove(local+name+'process/geno-grm-'+snp+'.txt')
    
    os.rename('output/'+name[:-1]+'-grm-'+snp+'.'+grmParm+'XX.txt',local+name+'process/grm-'+snp+'.txt')
        
    DBUpload(name+'process/grm-'+snp+'.txt',parms,toPickle=False)

    return()