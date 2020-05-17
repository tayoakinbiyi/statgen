import numpy as np
import pdb
import time
import psutil

import ELL.init.classMethods as initClass
import ELL.preCompute.classMethods as preComputeClass
import ELL.score.classMethods as scoreClass

class ell:    
    __init__=initClass.__init__
    preCompute=preComputeClass.preCompute
    minMaxLamPerKInitial = preComputeClass.minMaxLamPerKInitial
    minMaxLamPerKFinal = preComputeClass.minMaxLamPerKFinal
    minMaxKPerBin = preComputeClass.minMaxKPerBin 
    callGetLamEllByK = preComputeClass.callGetLamEllByK
    callGetLamEllByK=preComputeClass.callGetLamEllByK
    
    score=scoreClass.score
