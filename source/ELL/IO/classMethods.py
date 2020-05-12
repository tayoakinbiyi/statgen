import pdb
import numpy as np
import os
import subprocess

def save(self):
    subprocess.call(['mkdir','-p','lamEllByK'])
    try:
        np.savetxt('lamEllByK/lamEllByK',self.lamEllByK,delimiter='\t')
    except:
        pass
    
    try:
        np.savetxt('lamEllByK/ellGrid',self.ellGrid,delimiter='\t')
    except:
        pass
    
    try:
        np.savetxt('lamEllByK/offDiag',self.offDiag,delimiter='\t')
    except:
        pass

    try:
        np.savetxt('lamEllByK/ref',self.ref,delimiter='\t')
    except:
        pass
            
    return()

def load(self):
    try:
        self.lamEllByK=np.loadtxt('lamEllByK/lamEllByK',delimiter='\t')
    except:
        pass
    
    try:
        self.ellGrid=np.loadtxt('lamEllByK/ellGrid',delimiter='\t')
    except:
        pass
    
    try:
        self.offDiag=np.loadtxt('lamEllByK/offDiag',delimiter='\t')
        oLen=len(self.offDiag)
        self.N=int(1/2+np.sqrt(1/4+2*oLen))    
    except:
        pass

    try:
        self.ref=np.loadtxt('lamEllByK/ref',delimiter='\t')
    except:
        pass
        
    return()
