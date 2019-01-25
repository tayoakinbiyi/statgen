import pdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def strConcat(df):
    minDF=df.power.rank(ascending=False,method='min').astype('int').astype(str)
    maxDF=df.power.rank(ascending=False,method='max').astype('int').astype(str)
    concat=pd.Series(minDF.astype('int').astype(str),name='r')
    sel=minDF!=maxDF
    concat[sel]=minDF[sel]+'-'+maxDF[sel]
    return(pd.concat([df,concat],axis=1))

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
    power=power.groupby(['mu','eps'],sort=False).apply(strConcat).sort_values(by=['mu','eps','r'],ascending=[False,False,True])

    rawStats.to_csv(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-raw.csv')
    
    fail=rawStats[rawStats.Type.str.contains('fail')]
    failAvg=fail.groupby(['mu','eps','nullHyp','Type']).apply(lambda df:pd.DataFrame({'avgFailRate':100*np.mean(df.Value),
        'pctAllFail':100*np.mean(df.Value==1)},index=[0]).astype(int))
    failAvg=failAvg.reset_index().drop(columns='level_4')

    numType=len(power.Type.drop_duplicates().values.tolist())

    fig, axs = plt.subplots(numType+3, 1,tight_layout=True)
    fig.set_figwidth(delta/2,forward=True)
    fig.set_figheight(delta*(numType+3)/2,forward=True)

    axs=axs.flatten()

    j=0
    j=heatMap(power,'power','r','power','eps','mu',0,100,j,axs) 
    j=heatMap(failAvg,'','pctAllFail','avgFailRate','eps','mu',0,100,j,axs)
    
    fig.savefig(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'.png')

def heatMap(df,name,textCol,colorCol,index,columns,vmin,vmax,j,axs):
    for Type in df.Type.drop_duplicates().values.tolist():
        cols=df[columns].drop_duplicates()
        rows=df[index].drop_duplicates()
        im=df[df.Type==Type].pivot(values=colorCol,index=index,columns=columns).fillna(0).astype('int').values
        textDF=df[df.Type==Type].astype(str).pivot(values=textCol,index=index,columns=columns).fillna('').values
        im = axs[j].imshow(im, interpolation='nearest', cmap='Greys',vmin=vmin,vmax=vmax)
        
        axs[j].set_xticks(np.arange(len(textDF)))
        axs[j].set_yticks(np.arange(len(textDF)))
        axs[j].set_xticklabels(cols,fontsize=10,rotation=-30)
        axs[j].set_yticklabels(rows,fontsize=10)
        axs[j].set_xlabel('mu')
        axs[j].set_ylabel('eps')
        axs[j].tick_params(axis='x',pad=7)

        cbar=plt.colorbar(im, ax=axs[j],cmap='Greys')
        cbar.set_label(colorCol)

        for x in range(len(textDF)):
            for y in range(len(textDF)):
                axs[j].text(y, x, textDF[x, y], ha="center", va="center", color="r",fontsize=10)
                
        axs[j].set_title(Type+'-'+name)
        j+=1

    return(j)
