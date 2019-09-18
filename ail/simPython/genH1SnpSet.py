from ail.opPython.DB import *
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cut_tree
import matplotlib.pyplot as plt

# 2)
def genH1SnpSet(parms):
    H1Chr=parms['H1Chr']
    local=parms['local']
    name=parms['name']
    treeHeights=parms['treeHeights']
    N=DBRead(name+'process/N',parms,toPickle=True)[0]
    
    linkageMethods=['average','complete','weighted']

    if not DBIsFile(name+'sim','H1SnpSet',parms):
        allIds=DBRead(name+'process/allIds',parms,toPickle=True)
        mouseIds=DBRead(name+'process/mouseIds',parms,toPickle=True)
        
        H1SnpSet=pd.read_csv(local+'data/'+parms['snpFile'],sep='\t',header=None,index_col=None)
        H1SnpSet.columns=['chr','Mbp','minor','major'] +allIds
        H1SnpSet=H1SnpSet.loc[H1SnpSet['chr'].values.flatten()==H1Chr]
        H1SnpSet=H1SnpSet.loc[:,['Mbp','minor','major']+mouseIds]
        
        DBWrite(H1SnpSet,name+'sim/H1SnpSet',parms,toPickle=True)
    else:
        H1SnpSet=DBRead(name+'sim/H1SnpSet',parms,toPickle=True)
    
    if not DBIsFile(name+'process','H1SnpSet-Trees',parms):
        trees={}
        for linkageMethod in linkageMethods:
            links=linkage(H1SnpSet.iloc[:,3:].values, linkageMethod, 'correlation')

            fig,axs=plt.subplots(1,1,dpi=60)
            fig.set_figwidth(N/7,forward=True)
            fig.set_figheight(5,forward=True)
            den=dendrogram(links, color_threshold=0,ax=axs)
            axs.tick_params(axis='X',labelsize=20)
            fig.savefig(local+name+'plots/H1SnpSet-'+linkageMethod+'.png',bbox_inches='tight')
            DBUpload(name+'plots/H1SnpSet-'+linkageMethod+'.png',parms,toPickle=False)
            plt.close()

            trees[linkageMethod]=[]
            for i in range(len(treeHeights)):
                trees[linkageMethod]+=[pd.DataFrame({'loc':range(len(H1SnpSet)),'tree':cut_tree(links,
                    height=treeHeights[i]).flatten()}).groupby('tree').min()]

            treeSizes=[x.shape[0] for x in trees[linkageMethod]]

            pd.DataFrame({'treeHeight':treeHeights,'numClusters':treeSizes}).to_csv(local+name+
                'plots/H1SnpSet-'+linkageMethod+'.csv',index=False)
            DBUpload(name+'plots/H1SnpSet-'+linkageMethod+'.csv',parms,toPickle=False)
        DBWrite(trees,name+'process/H1SnpSet-Trees',parms,toPickle=True)
    else:
        trees=DBRead(name+'process/H1SnpSet-Trees',parms,toPickle=True)
    
    method=int(input(str({loc:linkageMethods[loc] for loc in range(len(linkageMethods))})+'  '))
    tree=trees[linkageMethods[method]][int(input('tree : '))]
    H1SnpSet=H1SnpSet.iloc[tree.values.flatten(),:].to_csv(local+name+'process/H1SnpSet.txt',sep='\t',index=False,header=False)        
    DBUpload(name+'process/H1SnpSet.txt',parms,toPickle=False)
    
    return()
