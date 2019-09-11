from ail.opPython.DB import *
import pandas as pd

# 2)
def genH1SnpSet(parms):
    H1Chr=parms['H1Chr']
    local=parms['local']
    name=parms['name']
    
    allIds=DBRead(name+'process/allIds',parms,toPickle=True)
    mouseIds=DBRead(name+'process/mouseIds',parms,toPickle=True)

    H1SnpSet=pd.read_csv(local+'data/'+parms['snpFile'],sep='\t',header=None,index_col=None)
    H1SnpSet.columns=np.append(['chr','Mbp','minor','major'],allIds)
    H1SnpSet=H1SnpSet[H1SnpSet['chr']==H1Chr]
    H1SnpSet=H1SnpSet[mouseIds]

    links=linkage(H1SnpSet.values, method, 'correlation')
    tree=pd.DataFrame({'loc':range(len(H1SnpSet)),'tree':cut_tree(links,height=.4)}).groupby('tree').min()
    
    H1SnpSet=H1SnpSet.iloc[tree['loc'],:].to_csv(local+name+'process/H1SnpSet.txt',sep='\t',index=False,header=False)        
    DBUpload(name+'process/H1SnpSet.txt',parms,toPickle=False)
    
    return()
