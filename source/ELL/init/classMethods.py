import numpy as np
from ELL.util import *
import ray
import pdb
from multiprocessing import cpu_count

def __init__(self,dList,offDiag,numCores=cpu_count(),reportMem=False):
    oLen=len(offDiag)
    self.N=int(1/2+np.sqrt(1/4+2*oLen))
    self.dList=(dList*self.N).astype(int)        
    self.reportMem=reportMem
    self.numCores=numCores
    self.offDiag=offDiag

    ray.init(num_cpus=self.numCores)

    if reportMem:
        memory('init')

        return
