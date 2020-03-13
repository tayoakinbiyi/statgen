import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from decimal import Decimal
from plotPython.myQQ import *
from plotPython.myHist import *
from zipfile import ZipFile

def plotZ(z,prefix=''):
    numSnps,numTraits=z.shape
    
    y=np.sort(np.mean(z,axis=0))
    x=norm.ppf(np.arange(1,numTraits+1)/(numTraits+1))/np.sqrt(numSnps)
    title='traitMean-theoretical'
    myQQ(x,y,prefix+title)

    y=np.sort(np.mean(z,axis=0))
    x=norm.ppf(np.arange(1,numTraits+1)/(numTraits+1))*np.std(y)
    title='traitMean-obs std'
    myQQ(x,y,prefix+title)

    y=np.sort(np.mean(z,axis=1))
    x=norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))/np.sqrt(numTraits)
    title='snpMean-theoretical'
    myQQ(x,y,prefix+title)

    y=np.sort(np.mean(z,axis=1))
    x=norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))*np.std(y)
    title='snpMean-obs std'
    myQQ(x,y,prefix+title)

    y=np.sort(z.flatten()**2)
    x=chi2.ppf(np.arange(1,len(y)+1)/(len(y)+1),1)
    title='z^2'
    myQQ(x,y,prefix+title)

    y=np.sort(np.mean(z**2,axis=0).flatten())
    x=1+np.std(y)*norm.ppf(np.arange(1,numTraits+1)/(numTraits+1))
    title='traitMean z^2 CLT std'
    myQQ(x,y,prefix+title)
    
    y=np.sort(np.mean(z**2,axis=0).flatten())
    x=chi2.ppf(np.arange(1,numTraits+1)/(numTraits+1),numSnps)/numSnps
    title='traitMean z^2 theoretical'
    myQQ(x,y,prefix+title)
    
    y=np.sort(np.mean(z**2,axis=1).flatten())
    x=1+np.std(y)*norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))
    title='snpMean z^2 CLT std'
    myQQ(x,y,prefix+title)

    y=np.sort(np.mean(z**2,axis=1).flatten())
    x=chi2.ppf(np.arange(1,numSnps+1)/(numSnps+1),numTraits)/numTraits
    title='snpMean z^2 theoretical'
    myQQ(x,y,prefix+title)

    myHist(np.corrcoef(z,rowvar=False)[np.triu_indices(numTraits,1)],'between trait')    
    myHist(np.corrcoef(z,rowvar=True)[np.triu_indices(numSnps,1)],'between snp')

    return()

