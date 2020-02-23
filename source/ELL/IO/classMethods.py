import ray
import pdb
import numpy as np
import os

def save(self):
    lamEllByK=self.lamEllByK
    ellGrid=self.ellGrid
    offDiag=self.offDiag
    
    np.savetxt('ref/lamEllByK',lamEllByK,delimiter='\t')
    np.savetxt('ref/ellGrid',ellGrid,delimiter='\t')
    np.savetxt('ref/offDiag',offDiag,delimiter='\t')
            
    return()

def load(self):
    if os.path.exists('ref/lamEllByK'):
        self.lamEllByK=np.loadtxt('ref/lamEllByK',delimiter='\t')
        self.ellGrid=np.loadtxt('ref/ellGrid',delimiter='\t')
        self.offDiag=np.loadtxt('ref/offDiag',delimiter='\t')
        oLen=len(self.offDiag)
        self.N=int(1/2+np.sqrt(1/4+2*oLen))    
        return(True)
    else:
        return(False)
