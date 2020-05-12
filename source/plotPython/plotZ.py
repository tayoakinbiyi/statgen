import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from decimal import Decimal
from plotPython.myQQ import *
from plotPython.myHist import *
from zipfile import ZipFile

def plotZ(z,title=''):
    numSnps,numTraits=z.shape
    
    y=np.sort(np.mean(z,axis=0))
    x=norm.ppf(np.arange(1,numTraits+1)/(numTraits+1))/np.sqrt(numSnps)
    myQQ(x,y,title,'traitMean')

    y=np.sort(np.mean(z,axis=1))
    x=norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))/np.sqrt(numTraits)
    myQQ(x,y,title,'snpMean')

    y=np.sort(np.mean(z**2,axis=0).flatten())
    x=chi2.ppf(np.arange(1,numTraits+1)/(numTraits+1),numSnps)/numSnps
    myQQ(x,y,title,'traitMean z^2')
    
    y=np.sort(np.mean(z**2,axis=1).flatten())
    x=chi2.ppf(np.arange(1,numSnps+1)/(numSnps+1),numTraits)/numTraits
    myQQ(x,y,title,'snpMean z^2')

    return()

