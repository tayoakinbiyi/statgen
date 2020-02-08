import ray
import pdb
import numpy as np

def save(self):
    lamEllByK=self.lamEllByK
    ellGrid=self.ellGrid
    offDiag=self.offDiag
    
    np.savetxt('lamEllByK',lamEllByK,delimiter='\t')
    np.savetxt('ellGrid',ellGrid,delimiter='\t')
    np.savetxt('offDiag',offDiag,delimiter='\t')
            
    return()

def load(self):
    self.lamEllByK=np.loadtxt('lamEllByK',delimiter='\t')
    self.ellGrid=np.loadtxt('ellGrid',delimiter='\t')
    self.offDiag=np.loadtxt('offDiag',delimiter='\t')
    oLen=len(offDiag)
    self.N=int(1/2+np.sqrt(1/4+2*oLen))    
    
    return()