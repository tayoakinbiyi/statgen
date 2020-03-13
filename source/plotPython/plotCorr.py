import pdb
import matplotlib.pyplot as plt

def plotCorr(corr,name):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(50,forward=True)
    fig.set_figheight(50,forward=True)
    axs.imshow(corr,interpolation='nearest', cmap='seismic',vmin=-1,vmax=1)
    fig.savefig('diagnostics/'+name+'.png',bbox_inches='tight')
    plt.close()    
    return()