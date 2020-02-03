import numpy as np
from ELL.util import *
import ray

def __init__(self,N,dList,numCores,reportMem=False):
        self.N=N
        self.dList=dList.astype(int)        
        self.reportMem=reportMem
        self.numCores=numCores
        
        ray.init(num_cpus=self.numCores)
        
        if reportMem:
            memory('init')
        
        return
