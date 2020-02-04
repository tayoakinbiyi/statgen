import ray
import pdb
import numpy as np

def save(self):
    lamEllByK=self.lamEllByK
    ellGrid=self.ellGrid
    
    np.savetxt('lamEllByK',lamEllByK,delimiter='\t')
    np.savetxt('ellGrid',ellGrid,delimiter='\t')
            
    return()

def load(self):
    self.lamEllByK=np.loadtxt('lamEllByK',delimiter='\t')
    self.ellGrid=np.loadtxt('ellGrid',delimiter='\t')
    
    return()