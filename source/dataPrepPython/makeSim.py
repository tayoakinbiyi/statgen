import pandas as pd
import numpy as np
import pdb
import pyreadr
import subprocess
import shutil
import random
from scipy.stats import norm

from opPython.DB import *
from genPython.makePSD import *

from dataPrepPython.writeGrm import *
from dataPrepPython.writeSnps import *
from dataPrepPython.writeY import *

def makeSim(parms,genSnps=True,genGrm=True,genY=True,genCov=True):
    print('simSetup')
    
    local=parms['local']
    numCores=parms['numCores']
    response=parms['response']
    
    numSubjects=parms['parms'][1]
    numSnps=parms['parms'][3]
                                  
    if genSnps:
        DBCreateFolder('snps',parms)        
        writeSnps(parms)

    if genGrm:
        DBCreateFolder('grm',parms)
        writeGrm(parms,1)    

        for snp in range(2,len(numSnps)+1):
            subprocess.call(['ln','-s','fast-', 'grm/fast-'+str(snp)])
            subprocess.call(['ln','-s','gemma-1', 'grm/gemma-'+str(snp)])
            subprocess.call(['ln','-s','grm-1', 'grm/grm-'+str(snp)])
            subprocess.call(['ln','-s','eigh-1', 'grm/eigh-'+str(snp)])
    
    if genY:
        DBCreateFolder('Y',parms)
        writeY(parms)

    if genCov:
        DBCreateFolder('cov',parms)
        np.savetxt('cov/cov.phe',np.array([[0,id_,1] for id_ in range(numSubjects)]),delimiter='\t',fmt='%s')
        np.savetxt('cov/cov.txt',np.ones([numSubjects,1]),delimiter='\t',fmt='%s')

    return()
