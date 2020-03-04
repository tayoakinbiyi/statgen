import numpy as np
from ELL.util import *
import pdb
from multiprocessing import cpu_count

def __init__(self,dList,N,numCores=cpu_count(),reportMem=True):
    assert np.max(dList)<=.5
    
    self.N=N
    self.dList=(dList*self.N).astype(int)        
    self.reportMem=reportMem
    self.numCores=numCores

    if reportMem:
        memory('init')

    return
