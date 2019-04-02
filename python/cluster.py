import scipy.stats as st
import numpy as np
import pdb
import matplotlib.pyplot as pl
import pandas as pd
from math import log
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def cluster(fileName,datName,N=None):    
    data=pd.read_csv(fileName,sep=',')
    if N==None:
        N=data.shape[1]
    data=data.iloc[:,0:N]
    data=np.cov(data,rowvar=False)
    diag=np.sqrt(np.diag(data).reshape(-1,1))
    data=(data/np.matmul(np.abs(diag),np.abs(diag).T))
    
    dissimilarity = 1 - np.abs(data)
    np.fill_diagonal(dissimilarity,0)
    hierarchy = linkage(squareform(dissimilarity), method='average')
    labels = fcluster(hierarchy, 0.5, criterion='distance')    

    upp=np.triu_indices(N,1)

    hist=data[upp].flatten()
    hist=hist[np.abs(hist)<np.percentile(np.abs(hist),99)].tolist()
    
    name=str(N)+'-'+datName
    pl.hist(hist,density=False,bins='sturges')
    pl.title(name)
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(name+"-precluster-hist.png")  
    
    sOrd=np.argsort(labels)
    ordData=np.abs(data[sOrd,sOrd])
    
    fig,axs=pl.subplots(1,1)    
    fig.set_figwidth(N/2,forward=True)
    fig.set_figheight(N/2,forward=True)
    
    im = axs.imshow(data, interpolation='nearest', cmap='Greys')

    pl.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False) # labels along the bottom edge are off
    
    pl.colorbar(im, ax=axs,cmap='Greys')
    fig.savefig(str(N)+'-'+datName+'-postcluster-heatmap.png')
    
def hcluster():
    N=1200
    data=pd.read_csv('mouse.csv',sep=',').T.values[0:N,:]
    data=data[np.array([len(np.unique(x))>2 for x in data])]    
    Z=linkage(data, 'single', 'correlation')
    den=dendrogram(Z, color_threshold=0)
    
    leaves=den['leaves']
    icoord=den['icoord']
    dcoord=den['dcoord']
    
    
    axs=plt.gca()
    axs.tick_params(axis='X',labelsize=20)
    fig=plt.gcf()
    fig.set_figwidth(N/3,forward=True)
    fig.set_figheight(5,forward=True)
    fig.savefig('full_dendogram.png',bbox_inches='tight')
    pdb.set_trace()
    
if __name__ == '__main__':
    hcluster()
    #cluster('rat.csv','rat',300)
    
    