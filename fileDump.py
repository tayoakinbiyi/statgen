import pdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def fileDump(parms):
    j=0
    rawStats=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1
    delta=parms[j];j+=1
    
    nonFail=rawStats[~rawStats.Type.str.contains('fail')]
    
    alpha=nonFail[nonFail.nullHyp].groupby(['mu','eps','nullHyp','Type'],sort=False).apply(
        lambda df:np.nanpercentile(df.Value,q=95))
    alpha.name='alpha'
    alpha=alpha.reset_index()
    
    power=nonFail[~nonFail.nullHyp].merge(alpha.drop(columns=['nullHyp','mu','eps']),on=['Type']).groupby(
        ['mu','eps','nullHyp','Type']).apply(lambda df:100*np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index()
    power.drop(columns='nullHyp',inplace=True)
    
    power=power.groupby(['mu','eps'],sort=False).apply(lambda df:pd.concat([df,pd.Series(df.power.rank(ascending=False),
        name='r')],axis=1)).sort_values(by=['mu','eps','r'],ascending=[False,False,True])

    rawStats.to_csv(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-raw.csv')
    
    fail=rawStats[rawStats.Type.str.contains('fail')]
    failAvg=fail.groupby(['mu','eps','nullHyp','Type']).apply(lambda df:pd.DataFrame({'avg':100*np.mean(df.Value),
        'all':100*np.mean(df.Value==1)},index=[0]))
    failAvg=failAvg.reset_index().drop(columns='level_4')

    numType=len(power.Type.drop_duplicates().values.tolist())

    fig, axs = plt.subplots(2*numType+6, 1,tight_layout=True)
    fig.set_figwidth(delta/2,forward=True)
    fig.set_figheight(delta*(2*numType+6)/2,forward=True)

    axs=axs.flatten()

    j=0
    j=heatMap(power,'power','power','eps','mu',0,100,j,axs) 
    j=heatMap(power,'rank','r','eps','mu',1,numType,j,axs)   
    j=heatMap(failAvg,'avgFail','avg','eps','mu',0,100,j,axs)
    j=heatMap(failAvg,'pctAllFail','all','eps','mu',0,100,j,axs)
    
    fig.savefig(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'.png')

def heatMap(df,name,values,index,columns,vmin,vmax,j,axs):
    for Type in df.Type.drop_duplicates().values.tolist():
        t_df=pd.pivot_table(df[df.Type==Type],values=values,index=index,columns=columns).fillna(0)
        data=t_df.values.astype('int')
        
        im = axs[j].imshow(data, interpolation='nearest', cmap='Greys',vmin=vmin,vmax=vmax)
        
        axs[j].set_xticks(np.arange(len(t_df)))
        axs[j].set_yticks(np.arange(len(t_df)))
        axs[j].set_xticklabels(t_df.columns,fontsize=10,rotation=-30)
        axs[j].set_yticklabels(t_df.index,fontsize=10)
        axs[j].set_xlabel('mu')
        axs[j].set_ylabel('eps')
        axs[j].tick_params(axis='x',pad=7)

        plt.colorbar(im, ax=axs[j],cmap='Greys')

        for x in range(len(t_df)):
            for y in range(len(t_df)):
                text = axs[j].text(y, x, data[x, y], ha="center", va="center", color="r",fontsize=10)
                
        axs[j].set_title(Type+'-'+name)
        j+=1

    return(j)
