import pdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import gc

def bestFunc(df,H1,H01):
    mpPool=(df.power.max()+df.power)/2000
    mmPool=(df.power.max()+df.power.min())/2000
    Types=df.Type[((df.power.max()-df.power)/1000-1.96*np.sqrt(2*mpPool*(1-mpPool)/H1)<=0)&(
        (df.power/1000-0.05)-1.96*np.sqrt(2*.05*(1-.05)/H1)>0)]
    return(pd.DataFrame({'Type':'\n'.join(Types)+'\n'+str(df.power.max().astype(int)) if len(Types)<len(df) else
        str(df.power.max().astype(int)),'len':len(Types)+1 if len(Types)<len(df) else 1},index=[0]))  

def fileDump(dat):
    heatMapPower(dat[0],dat[2]) 
    heatMapFail(dat[1],dat[2]) 

def heatMapPower(power,parms):
    if not parms['plot']:
        return()

    N=parms['N']
    H0=parms['H0']
    H1=parms['H1']
    H01=parms['H01']
    sigName=parms['sigName']  
    fontsize=parms['fontsize']
    
    Types=np.sort(power.Type.drop_duplicates().sort_values().values.flatten())
                    
    mu=np.round(sorted(power.mu.drop_duplicates().values.tolist()),2)
    eps=np.round(sorted(power.eps.drop_duplicates().values.tolist()),2)
    title=['Power','Rank','Fraction of Max']

    mat=[0]*len(Types)
    for Type in range(len(Types)):
        mat[Type]=power[power.Type==Types[Type]].pivot(values='power',index='eps',columns='mu').fillna(500).astype(int).values
    
    mMax=power.groupby(['eps','mu'],sort=False)['power'].max().reset_index().pivot(
        values='power',index='eps',columns='mu').fillna(500).astype(int).values
    mMin=power.groupby(['eps','mu'],sort=False)['power'].min().reset_index().pivot(
        values='power',index='eps',columns='mu').fillna(500).astype(int).values
  
    fig, axs = plt.subplots(len(Types),3,dpi=50)   
    fig.set_figwidth(len(mu)*3,forward=True)
    fig.set_figheight(len(Types)*len(eps)*1.5,forward=True)
    
    textDF=[0,0,0]   
        
    for Type in range(len(Types)):
        textDF[0]=mat[Type].astype(str)
        textDF[0][0,0]=str(int(1000*(mat[Type][0,0]/1000-1.96*np.sqrt((.05*.95)/H01))))+'-'+str(int(1000*(mat[Type][0,0]/1000+
            1.96*np.sqrt((.05*.95)/H01))))
        textDF[1]=power[power.Type==Types[Type]].pivot(values='r',index='eps',columns='mu').fillna(0).values
        textDF[2]=(1000*mat[Type]/mMax).astype(int)

        axs[Type,0].imshow(mat[Type],interpolation='nearest', cmap='seismic',vmin=0,vmax=1000)
        axs[Type,1].imshow(mat[Type],interpolation='nearest', cmap='seismic',vmin=0,vmax=1000)
        axs[Type,2].imshow(textDF[2],interpolation='nearest', cmap='Greys',vmin=0,vmax=1000)

        for Plot in range(len(textDF)):
            axs[Type,Plot].set_xticks(np.arange(mat[Type].shape[1]))
            axs[Type,Plot].set_yticks(np.arange(mat[Type].shape[0]))
            axs[Type,Plot].set_xticklabels(mu,fontsize=fontsize,rotation=-30)
            axs[Type,Plot].set_yticklabels(eps,fontsize=fontsize)
            axs[Type,Plot].set_xlabel('mu',fontsize=fontsize)
            axs[Type,Plot].set_ylabel('eps',fontsize=fontsize)
            axs[Type,Plot].tick_params(axis='x',pad=7)
            axs[Type,Plot].set_title(sigName+' , '+Types[Type]+' , '+title[Plot],fontsize=fontsize)
            
            for x in range(len(eps)):
                for y in range(len(mu)):
                    axs[Type,Plot].text(y, x, textDF[Plot][x,y], ha="center", va="center", color="black",fontsize=fontsize)              

    fig.savefig('../heatmap/heatmap-power-N:'+str(N)+'-H0:'+str(H0)+'-H1:'+str(H1)+'-H01:'+str(H01)+'-Sig:'+sigName+'.png',
                bbox_inches='tight')

    bestType=power.groupby(['mu','eps'],sort=False).apply(bestFunc,H1,H01).reset_index()
    lenType=bestType.pivot(values='len',index='eps',columns='mu').fillna(1).values
    textType=bestType.pivot(values='Type',index='eps',columns='mu').fillna('').values
    textType[0,0]=str(mMin[0,0])+'-'+str(mMax[0,0])

    fig, axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(len(mu)*2,forward=True)
    fig.set_figheight(lenType.sum(axis=0).max()*2,forward=True)
       
    axs.imshow(mMax,interpolation='nearest', cmap='seismic',vmin=0,vmax=1000)
    
    title=['Best Power','Best Stat']
    axs.set_xticks(np.arange(textType.shape[1]))
    axs.set_yticks(np.arange(textType.shape[0]))
    axs.set_xticklabels(mu,fontsize=fontsize,rotation=-30)
    axs.set_yticklabels(eps,fontsize=fontsize)
    axs.set_xlabel('mu',fontsize=fontsize)
    axs.set_ylabel('eps',fontsize=fontsize)
    axs.tick_params(axis='x',pad=7)
    axs.set_title(sigName+' , '+title[0],fontsize=2*fontsize)

    for x in range(len(eps)):
        for y in range(len(mu)):
            axs.text(y, x, textType[x,y], ha="center", va="center", color="black",fontsize=1.2*fontsize)              
    
    fig.savefig('../heatmap/heatmap-best-N:'+str(N)+'-H0:'+str(H0)+'-H1:'+str(H1)+'-H01:'+
                str(H01)+'-Sig:'+sigName+'.png',bbox_inches='tight')

def heatMapFail(fail,parms):    
    return()
    if not parms['plot']:
        return()

    N=parms['N']
    H0=parms['H0']
    H1=parms['H1']
    H01=parms['H01']
    sigName=parms['sigName']  
    fontsize=parms['fontsize']
    
    if len(fail)==0:
        return()
    
    Types=np.sort(fail.Type.drop_duplicates().values.flatten())
       
    mu=np.round(sorted(fail.mu.drop_duplicates().values.tolist()),2)
    eps=np.round(sorted(fail.eps.drop_duplicates().values.tolist()),2)
    title=['Average over runs: % of k Approx Failed','% of runs: Approx Failed on All k']

    avgFailRate=[0]*len(Types)
    pctAllFail=[0]*len(Types)
    for Type in range(len(Types)):
        avgFailRate[Type]=fail[fail.Type==Types[Type]].pivot(values='avgFailRate',index='eps',columns='mu').fillna(0).astype(int).values
        pctAllFail[Type]=fail[fail.Type==Types[Type]].pivot(values='pctAllFail',index='eps',columns='mu').fillna(0).astype(int).values
    
    #pdb.set_trace()
    fig, axs = plt.subplots(len(Types),2,dpi=50)   
    fig.set_figwidth(len(mu)*3,forward=True)
    fig.set_figheight(len(Types)*len(eps)*1.5,forward=True)

    textDF=[0,0]   
    for Type in range(len(Types)):
        textDF[0]=fail[fail.Type==Types[Type]].pivot(values='avgFailRate',index='eps',columns='mu').fillna(0).astype(int).values
        textDF[1]=fail[fail.Type==Types[Type]].pivot(values='pctAllFail',index='eps',columns='mu').fillna(0).astype(int).values

        axs[Type,0].imshow(textDF[0],interpolation='nearest', cmap='seismic',vmin=500,vmax=1000)
        axs[Type,1].imshow(textDF[1],interpolation='nearest', cmap='seismic',vmin=500,vmax=1000)

        for Plot in range(len(textDF)):
            axs[Type,Plot].set_xticks(np.arange(textDF[0].shape[1]))
            axs[Type,Plot].set_yticks(np.arange(textDF[0].shape[0]))
            axs[Type,Plot].set_xticklabels(mu,fontsize=fontsize,rotation=-30)
            axs[Type,Plot].set_yticklabels(eps,fontsize=fontsize)
            axs[Type,Plot].set_xlabel('mu',fontsize=fontsize)
            axs[Type,Plot].set_ylabel('eps',fontsize=fontsize)
            axs[Type,Plot].tick_params(axis='x',pad=7)
            axs[Type,Plot].set_title(sigName+' , '+Types[Type]+' , '+title[Plot],fontsize=fontsize)
            
            for x in range(len(eps)):
                for y in range(len(mu)):
                    axs[Type,Plot].text(y, x, textDF[Plot][x,y], ha="center", va="center", color="black",fontsize=fontsize)              

    fig.savefig('heatmap-fail-N:'+str(N)+'-H0:'+str(H0)+'-H1:'+str(H1)+'-H01:'+str(H01)+'-Sig:'+sigName+'.png',bbox_inches='tight')
    
def nPlot(power,H1,sigName):
    N=power['N'].drop_duplicates().sort_values().values
    mu=power['mu'].drop_duplicates().sort_values().values.round(3)
    mu=mu[mu!=0]
    Types=power.Type.drop_duplicates().values
    
    fig, axs = plt.subplots(len(N),len(mu),dpi=50)   
    fig.set_figwidth(len(mu)*10,forward=True)
    fig.set_figheight(len(N)*10,forward=True)

    for t_N in range(len(N)):
        for t_mu in range(len(mu)):
            for Type in Types:                
                df=power[(power.mu==mu[t_mu])&(power.N==N[t_N])&(power.Type==Type)].sort_values(by='eps').plot(x='eps',y='power',
                    ax=axs[t_N,t_mu],label=Type,linewidth=6,fontsize=20)
                axs[t_N,t_mu].set_xlabel('eps',fontsize=20)
                axs[t_N,t_mu].set_ylabel('power',fontsize=20)
                axs[t_N,t_mu].legend(fontsize=20)
                axs[t_N,t_mu].set_title(Type+'-N='+str(N[t_N])+'-mu='+str(mu[t_mu]),fontsize=20)

                
    fig.savefig('N-plot-H1:'+str(H1)+'-'+sigName+'.png',bbox_inches='tight')
                