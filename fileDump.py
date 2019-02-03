import pdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as plt
import gc

def strConcat(df):
    minDF=df.power.rank(ascending=False,method='min').astype('int').astype(str)
    maxDF=df.power.rank(ascending=False,method='max').astype('int').astype(str)
    concat=pd.Series(minDF.astype('int').astype(str),name='r')
    sel=minDF!=maxDF
    concat[sel]=minDF[sel]+'-'+maxDF[sel]
    return(pd.concat([df,concat],axis=1))

def fileDump(parms,Types=None):
    j=0
    rawStats=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1
    #pdb.set_trace()   
    #rawStats=pd.read_csv(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-raw.csv')
    '''fail=rawStats[rawStats.Type.str.contains('fail')]
    failAvg=fail.groupby(['mu','eps','nullHyp','Type']).apply(lambda df:pd.DataFrame({'avgFailRate':100*np.mean(df.Value),
        'pctAllFail':100*np.mean(df.Value==1)},index=[0]).astype(int))
    failAvg=failAvg.reset_index().drop(columns='level_4')'''

    heatMapPower(parms,Types) 
    #j=heatMap(failAvg,'','pctAllFail','avgFailRate','eps','mu',0,100,j,axs) 

def heatMapPower(parms,Types):
    j=0
    rawStats=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1    
    
    if Types is None:
        Types=np.sorted(rawStats.Type.drop_duplicates().values.flatten())
    else:
        rawStats=rawStats[rawStats.Type.isin(Types)]
    
    nonFail=rawStats[~rawStats.Type.str.contains('fail')]    
    
    alpha=nonFail[nonFail.mu*nonFail.eps==0].groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
    alpha.name='alpha'
    alpha=alpha.reset_index()
    
    power=nonFail[nonFail.mu*nonFail.eps>0].merge(alpha,on=['Type'])
    power=power.groupby(['mu','eps','Type']).apply(lambda df:1000*np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index()
    power=power.groupby(['mu','eps'],sort=False).apply(strConcat).sort_values(by=['mu','eps','r'],ascending=[False,False,True])
    
    mu=np.round(sorted(power.mu.drop_duplicates().values.tolist()),2)
    eps=np.round(sorted(power.eps.drop_duplicates().values.tolist()),2)

    mat=[0]*len(Types)
         
    for Type in range(len(Types)):
        mat[Type]=power[power.Type==Types[Type]].pivot(values='power',index='eps',columns='mu').fillna(0).astype('int').values
       
    for Type_x in range(len(Types)):
        fig, axs = plt.subplots(1,len(Types),tight_layout=True,dpi=50)   
        fig.set_figwidth(len(mu)*len(Types),forward=True)
        fig.set_figheight(len(eps)+2,forward=True)

        for Type_y in range(len(Types)):
            axs[Type_y].set_xticks(np.arange(mat[Type_x].shape[1]))
            axs[Type_y].set_yticks(np.arange(mat[Type_x].shape[0]))
            axs[Type_y].set_xticklabels(mu,fontsize=20,rotation=-30)
            axs[Type_y].set_yticklabels(eps,fontsize=20)
            axs[Type_y].set_xlabel('mu',fontsize=20)
            axs[Type_y].set_ylabel('eps',fontsize=20)
            axs[Type_y].tick_params(axis='x',pad=7)

            axs[Type_y].set_title(Types[Type_x]+' / '+Types[Type_y],fontsize=20)
            if Type_y==Type_x:
                textDF=power[power.Type==Types[Type_x]].pivot(values='r',index='eps',columns='mu').values
                axs[Type_y].imshow(mat[Type_x],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)
            else:
                textDF=(1000*mat[Type_x]/(mat[Type_x]+mat[Type_y])).astype(int)
                axs[Type_y].imshow(textDF,interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)

            for x in range(len(eps)):
                for y in range(len(mu)):
                    axs[Type_y].text(y, x, textDF[x,y], ha="center", va="center", color="r",fontsize=20)              

        fig.savefig('power-'+Types[Type_x]+'-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'.png')
    
