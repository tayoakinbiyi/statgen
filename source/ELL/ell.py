import ray
import numpy as np
import pdb
import time
import psutil

import ELL.init.classMethods as initClass
import ELL.fit.classMethods as fitClass
import ELL.score.classMethods as scoreClass
import ELL.monteCarlo.classMethods as monteCarloClass
import ELL.markov.classMethods as markovClass
import ELL.IO.classMethods as ioClass

class ell:    
    __init__=initClass.__init__
    fit=fitClass.fit
    minMaxLamPerKInitial = fitClass.minMaxLamPerKInitial
    minMaxLamPerKFinal = fitClass.minMaxLamPerKFinal
    minMaxKPerBin = fitClass.minMaxKPerBin 
    callLamEllByK = fitClass.callLamEllByK
    loopCallLamEllByK=fitClass.loopCallLamEllByK
    
    score=scoreClass.score

    markov=markovClass.markov
    
    monteCarlo=monteCarloClass.monteCarlo
    
    save=ioClass.save
    load=ioClass.load
