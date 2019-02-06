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

def fileDump(parms,Types=None,fontsize=20):
    heatMapPower(parms,Types,fontsize) 
    heatMapFail(parms,fontsize) 

def heatMapPower(parms,Types,fontsize):
    j=0
    power=parms[j];j+=1
    fail=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1    
    mu_delta=parms[j];j+=1
    eps_frac=parms[j];j+=1
    
    if Types is None:
        Types=np.sort(power.Type.drop_duplicates().values.flatten())
    else:
        power=power[power.Type.isin(Types)]
    
    alpha=power[power.mu*power.eps==0].groupby('Type',sort=False).apply(lambda df:np.nanpercentile(df.Value,q=95))
    alpha.name='alpha'
    alpha=alpha.reset_index()
    
    power=power[power.mu*power.eps>0].merge(alpha,on=['Type'])
    power=power.groupby(['mu','eps','Type']).apply(lambda df:1000*np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index()
    power=power.groupby(['mu','eps'],sort=False).apply(strConcat).sort_values(by=['mu','eps','r'],ascending=[False,False,True])
                    
    mu=np.round(sorted(power.mu.drop_duplicates().values.tolist()),2)
    eps=np.round(sorted(power.eps.drop_duplicates().values.tolist()),2)
    title=['-Power','-Rank','-Fraction of Max']

    mat=[0]*len(Types)
    for Type in range(len(Types)):
        mat[Type]=power[power.Type==Types[Type]].pivot(values='power',index='eps',columns='mu').fillna(0).astype(int).values
    mMax=power.groupby(['eps','mu'],sort=False)['power'].max().reset_index().pivot(
        values='power',index='eps',columns='mu').fillna(0).astype(int).values
    
    fig, axs = plt.subplots(len(Types)+1,3,dpi=50)   
    fig.set_figwidth(len(mu)*4,forward=True)
    fig.set_figheight(len(Types)*len(eps)*2,forward=True)

    textDF=[0,0,0]   
    for Type in range(len(Types)):
        textDF[0]=mat[Type]
        textDF[1]=power[power.Type==Types[Type]].pivot(values='r',index='eps',columns='mu').values
        textDF[2]=(1000*mat[Type]/mMax).astype(int)

        axs[Type,0].imshow(mat[Type],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)
        axs[Type,1].imshow(mat[Type],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)
        axs[Type,2].imshow(textDF[2],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)

        for Plot in range(len(textDF)):
            axs[Type,Plot].set_xticks(np.arange(mat[Type].shape[1]))
            axs[Type,Plot].set_yticks(np.arange(mat[Type].shape[0]))
            axs[Type,Plot].set_xticklabels(mu,fontsize=fontsize,rotation=-30)
            axs[Type,Plot].set_yticklabels(eps,fontsize=fontsize)
            axs[Type,Plot].set_xlabel('mu',fontsize=fontsize)
            axs[Type,Plot].set_ylabel('eps',fontsize=fontsize)
            axs[Type,Plot].tick_params(axis='x',pad=7)
            axs[Type,Plot].set_title(Types[Type]+title[Plot],fontsize=fontsize)
            
            for x in range(len(eps)):
                for y in range(len(mu)):
                    axs[Type,Plot].text(y, x, textDF[Plot][x,y], ha="center", va="center", color="r",fontsize=fontsize)              

    fig.savefig('heatmap-power-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.png')

    best=power[power.r.str.contains('[^0-9]*1[^0-9]*')].groupby(['mu','eps','power'],sort=False).apply(
        lambda df: pd.DataFrame({'Type':'\n'.join(df.Type),'len':df.shape[0]},index=[0])).reset_index()
    
    fig, axs = plt.subplots(2,1,dpi=50)   
    fig.set_figwidth(len(mu)*4,forward=True)
    fig.set_figheight(best.len.sum()*2,forward=True)
    
    textDF=[0,0]
    textDF[0]=best.pivot(values='power',index='eps',columns='mu').values
    textDF[1]=best.pivot(values='Type',index='eps',columns='mu').values
    
    axs[0].imshow(textDF[0],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)
    axs[1].imshow(textDF[0],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)
    
    title=['Best Power','Best Stat']
    for Plot in range(2):
        axs[Plot].set_xticks(np.arange(textDF[0].shape[1]))
        axs[Plot].set_yticks(np.arange(textDF[0].shape[0]))
        axs[Plot].set_xticklabels(mu,fontsize=fontsize,rotation=-30)
        axs[Plot].set_yticklabels(eps,fontsize=fontsize)
        axs[Plot].set_xlabel('mu',fontsize=fontsize)
        axs[Plot].set_ylabel('eps',fontsize=fontsize)
        axs[Plot].tick_params(axis='x',pad=7)
        axs[Plot].set_title(title[Plot],fontsize=2*fontsize)

        for x in range(len(eps)):
            for y in range(len(mu)):
                axs[Plot].text(y, x, textDF[Plot][x,y], ha="center", va="center", color="r",fontsize=1.2*fontsize)              
    
    fig.savefig('heatmap-best-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.png')

def heatMapFail(parms,fontsize):
    j=0
    power=parms[j];j+=1
    fail=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1    
    mu_delta=parms[j];j+=1
    eps_frac=parms[j];j+=1
    
    Types=np.sort(fail.Type.drop_duplicates().values.flatten())
       
    fail=fail.groupby(['mu','eps','Type']).apply(lambda df:pd.DataFrame({'avgFailRate':1000*np.mean(df.Value),
        'pctAllFail':1000*np.mean(df.Value==1)},index=[0]).astype(int)).reset_index().drop(columns='level_3')    
    
    mu=np.round(sorted(fail.mu.drop_duplicates().values.tolist()),2)
    eps=np.round(sorted(fail.eps.drop_duplicates().values.tolist()),2)
    title=['-Average Fail Rate','-Pct All FAil']

    avgFailRate=[0]*len(Types)
    pctAllFail=[0]*len(Types)
    for Type in range(len(Types)):
        avgFailRate[Type]=fail[fail.Type==Types[Type]].pivot(values='avgFailRate',index='eps',columns='mu').fillna(0).astype(int).values
        pctAllFail[Type]=fail[fail.Type==Types[Type]].pivot(values='pctAllFail',index='eps',columns='mu').fillna(0).astype(int).values
    
    fig, axs = plt.subplots(len(Types),2,dpi=50)   
    fig.set_figwidth(len(mu)*3,forward=True)
    fig.set_figheight(len(Types)*len(eps)*2,forward=True)

    textDF=[0,0]   
    for Type in range(len(Types)):
        textDF[0]=fail[fail.Type==Types[Type]].pivot(values='avgFailRate',index='eps',columns='mu').fillna(0).astype(int).values
        textDF[1]=fail[fail.Type==Types[Type]].pivot(values='pctAllFail',index='eps',columns='mu').fillna(0).astype(int).values

        axs[Type,0].imshow(textDF[0],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)
        axs[Type,1].imshow(textDF[1],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)

        for Plot in range(len(textDF)):
            axs[Type,Plot].set_xticks(np.arange(textDF[0].shape[1]))
            axs[Type,Plot].set_yticks(np.arange(textDF[0].shape[0]))
            axs[Type,Plot].set_xticklabels(mu,fontsize=fontsize,rotation=-30)
            axs[Type,Plot].set_yticklabels(eps,fontsize=fontsize)
            axs[Type,Plot].set_xlabel('mu',fontsize=fontsize)
            axs[Type,Plot].set_ylabel('eps',fontsize=fontsize)
            axs[Type,Plot].tick_params(axis='x',pad=7)
            axs[Type,Plot].set_title(Types[Type]+title[Plot],fontsize=fontsize)
            
            for x in range(len(eps)):
                for y in range(len(mu)):
                    axs[Type,Plot].text(y, x, textDF[Plot][x,y], ha="center", va="center", color="r",fontsize=fontsize)              

    fig.savefig('heatmap-fail-'+str(N)+'-'+str(H0)+'-'+str(H1)+'-'+sigName+'-'+str(mu_delta)+'-'+str(eps_frac)+'.png')
    