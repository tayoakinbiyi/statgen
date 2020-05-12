import numpy as np
import pdb
import time
import psutil

import ELL.init.classMethods as initClass
import ELL.preCompute.classMethods as preComputeClass
import ELL.score.classMethods as scoreClass
import ELL.monteCarlo.classMethods as monteCarloClass
import ELL.markov.classMethods as markovClass
import ELL.IO.classMethods as ioClass
import ELL.plot.classMethods as plotClass
import ELL.exact.classMethods as exactClass

class ell:    
    __init__=initClass.__init__
    preCompute=preComputeClass.preCompute
    minMaxLamPerKInitial = preComputeClass.minMaxLamPerKInitial
    minMaxLamPerKFinal = preComputeClass.minMaxLamPerKFinal
    minMaxKPerBin = preComputeClass.minMaxKPerBin 
    callGetLamEllByK = preComputeClass.callGetLamEllByK
    callGetLamEllByK=preComputeClass.callGetLamEllByK
    
    score=scoreClass.score

    markov=markovClass.markov
    
    monteCarlo=monteCarloClass.monteCarlo
    genRef=monteCarloClass.genRef
    monteCarloPval=monteCarloClass.monteCarloPval
    
    exact=exactClass.exact
    
    save=ioClass.save
    load=ioClass.load
    
    plot=plotClass.plot
