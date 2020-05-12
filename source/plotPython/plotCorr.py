import pdb
import matplotlib.pyplot as plt
from zipfile import ZipFile
import numpy as np

def plotCorr(corr,title):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(50,forward=True)
    fig.set_figheight(50,forward=True)
    axs.imshow(corr,interpolation='nearest', cmap='seismic',vmin=-1,vmax=1)
    with ZipFile('diagnostics/'+title+'.zip','w') as myZip:
        out=np.array(corr)
        myZip.writestr(title,'\n'.join(map(lambda x:'\t'.join(map(str,x)),out.tolist()))) 
    
    fig.savefig('diagnostics/'+title+'.png',bbox_inches='tight')
    plt.close()    
    return()