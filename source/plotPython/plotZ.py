import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from decimal import Decimal

def plotZ(z):
    numSnps,N=z.shape
    
    y=np.sort(np.mean(z,axis=0))
    x=norm.ppf(np.arange(1,N+1)/(N+1))/np.sqrt(numSnps)
    title='traitMean-theoretical std (ratio: %.2E' % Decimal(np.std(y)/np.sqrt(numSnps))+')'
    myQQ(x,y,title)

    y=np.sort(np.mean(z,axis=0))
    x=norm.ppf(np.arange(1,N+1)/(N+1))*np.std(y)
    title='traitMean-obs std'
    myQQ(x,y,title)

    y=np.sort(np.mean(z,axis=1))
    x=norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))/np.sqrt(N)
    title='snpMean-theoretical std'
    myQQ(x,y,title)

    y=np.sort(np.mean(z,axis=1))
    x=norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))*np.std(y)
    title='snpMean-obs std'
    myQQ(x,y,title)

    y=np.sort(z.flatten()**2)
    x=chi2.ppf(np.arange(1,len(y)+1)/(len(y)+1),1)
    title='z^2'
    myQQ(x,y,title)

    y=np.sort(np.mean(z**2,axis=0).flatten())
    x=1+np.std(y)*norm.ppf(np.arange(1,N+1)/(N+1))
    title='traitMean z^2 CLT std'
    myQQ(x,y,title)
    
    y=np.sort(np.mean(z**2,axis=0).flatten())
    x=chi2.ppf(np.arange(1,N+1)/(N+1),numSnps)/numSnps
    title='traitMean z^2 theoretical'
    myQQ(x,y,title)
    
    y=np.sort(np.mean(z**2,axis=1).flatten())
    x=1+np.std(y)*norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))     
    title='snpMean z^2 CLT std'
    myQQ(x,y,title)

    y=np.sort(np.mean(z**2,axis=1).flatten())
    x=chi2.ppf(np.arange(1,numSnps+1)/(numSnps+1),N)/N
    title='snpMean z^2 theoretical'
    myQQ(x,y,title)

    myHist(np.corrcoef(z,rowvar=False)[np.triu_indices(N,1)],'between trait')
    
    myHist(np.corrcoef(z,rowvar=True)[np.triu_indices(numSnps,1)],'between snp')

    return()

def myQQ(x,y,title):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    labMax=max(max(x),max(y))
    labMin=min(min(x),min(y))
    labMax+=.1*(labMax-labMin)
    labMin-=.1*(labMax-labMin)

    axs.set_xlim([labMin,labMax])
    axs.set_ylim([labMin,labMax])
    axs.scatter(x,y,c='.1')
    axs.plot([labMin,labMax], [labMin,labMax], ls="--", c=".3")  
    axs.set_xlabel('theoretical')
    axs.set_ylabel('observed')
    axs.set_title(title+' mu: '+('%.2E' % Decimal(np.mean(y)))+' std: '+('%.2E' % Decimal(np.std(y))))
    
    fig.savefig('diagnostics/'+title+'.png')
    plt.close('all') 

    return()

def myHist(x,title):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)

    axs.hist(x,bins=int(len(x)/8))
    axs.set_title(title+' mu: '+('%.2E' % Decimal(np.mean(x)))+' std: '+('%.2E' % Decimal(np.std(x))))
    
    fig.savefig('diagnostics/'+title+'.png')
    plt.close('all') 

    return()
