import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

def fileDump(parms):
    j=0
    rawStats=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1
    
    nonFail=rawStats[~rawStats.Type.str.contains('fail')]
    
    alpha=nonFail[nonFail.nullHyp].groupby(['mu','eps','nullHyp','Type'],sort=False).apply(
        lambda df:np.nanpercentile(df.Value,q=95))
    alpha.name='alpha'
    alpha=alpha.reset_index()
    
    power=nonFail[~nonFail.nullHyp].merge(alpha.drop(columns=['nullHyp','mu','eps']),on=['Type']).groupby(
        ['mu','eps','nullHyp','Type']).apply(lambda df:np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index()
    power.drop(columns='nullHyp',inplace=True)
    
    power=power.groupby(['mu','eps'],sort=False).apply(lambda df:pd.concat([df,pd.Series(df.power.rank(ascending=False),
        name='r')],axis=1)).sort_values(by=['mu','eps','r'],ascending=[False,False,True])

    numType=len(power.Type.drop_duplicates().values.tolist())

    fig, axs = plt.subplots(numType*4, 1,tight_layout=True)
    plt.rcParams["figure.figsize"]=[10,30*4*numType]
    axs=axs.flatten()

    j=0
    j=heatMap(power,'power','power','eps','mu',0,1,j,axs) 
    j=heatMap(power,'rank','r','eps','mu',0,1,j,axs)
   
    rawStats.to_csv(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-raw.csv')
    
    fail=rawStats[rawStats.Type.str.contains('fail')]
    failAvg=fail.groupby(['mu','eps','nullHyp','Type']).apply(lambda df:pd.DataFrame({'avg':np.mean(df.Value),
        'all':np.mean(df.Value==1)},index=[0]))
    failAvg.reset_index(inplace=True)

    j=heatMap(failAvg,'avgFail','avg','eps','mu',0,1,j,axs)
    j=heatMap(failAvg,'pctAllFail','all','eps','mu',0,1,j,axs)
    fig.savefig('t1.png')
    pdb.set_trace()

    fig.savefig(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'.pdf')

def heatMap(df,name,values,index,columns,vmin,vmax,j,axs):
    for Type in df.Type.drop_duplicates().values.tolist():
        t_df=pd.pivot_table(df[df.Type==Type],values=values,index=index,columns=columns)
        data=t_df.values
        im = axs[j].imshow(data, interpolation='nearest', aspect='auto',cmap='Greys')
        
        axs[j].set_xticks(np.arange(len(t_df)))
        axs[j].set_yticks(np.arange(len(t_df)))
        axs[j].set_xticklabels(t_df.columns,fontsize=10)
        axs[j].set_yticklabels(t_df.index,fontsize=10)

        for x in range(len(t_df)):
            for y in range(len(t_df)):
                text = axs[j].text(y, x, data[x, y], ha="center", va="center", color="r",fontsize=10)
                
        axs[j].set_title(Type+'-'+name)
        j+=1

    return(j)
