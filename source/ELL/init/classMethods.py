import numpy as np
from ELL.util import *
import pdb
from multiprocessing import cpu_count
from utility import *

def __init__(self,d,vZ,numCores=cpu_count()):    
    self.N=vZ.shape[1]
    self.nCr=nCr(vZ.shape[1])
    self.d=d
    self.numCores=numCores
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    self.offDiag=offDiag
    self.offDiagMeans=np.array([np.mean(offDiag), np.mean(offDiag**2), np.mean(offDiag**3),
        np.mean(offDiag**4),np.mean(offDiag**5), np.mean(offDiag**6), np.mean(offDiag**7),
        np.mean(offDiag**8),np.mean(offDiag**9), np.mean(offDiag**10)])
    self.L=makeL(vZ)    

    self.lamEllByK=None
    
    memory('init')

    return
