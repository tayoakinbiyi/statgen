import numpy as np
from ELL.util import *
import pdb
from multiprocessing import cpu_count
from utility import *

def __init__(self,d,N,vZ,numCores=cpu_count(),reportMem=True,qList=[.1,.05,.01,.001]):
    assert d<=.5*N
    
    self.N=N
    self.nCr=nCr(N)
    self.d=d
    self.reportMem=reportMem
    self.numCores=numCores
    self.qList=qList
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    self.offDiag=offDiag
    self.offDiagMeans=np.array([np.mean(offDiag), np.mean(offDiag**2), np.mean(offDiag**3),
        np.mean(offDiag**4),np.mean(offDiag**5), np.mean(offDiag**6), np.mean(offDiag**7),
        np.mean(offDiag**8),np.mean(offDiag**9), np.mean(offDiag**10)])
    self.L=makeL(vZ)    

    self.lamEllByK=None
    self.mcRef=None
    
    if reportMem:
        memory('init')

    return
