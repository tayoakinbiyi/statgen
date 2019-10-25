from ail.opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb

def makePVals(parms):
    name=parms['name']
    Types=parms['Types']
    numCores=parms['numCores']
    
    DBCreateFolder(name,'pvals',parms)
    
    statFiles=pd.Series(DBListFolder(name+'stats',parms),name='statFiles')
    IIDFiles=pd.Series(DBListFolder(name+'iid',parms),name='IIDFiles')
    nameParms=['mu:'+str(x[0])+'-eps:'+str(x[1]) for x in parms['muEpsRange']]

    iidStats={Type:[] for Type in Types}
    for Type in Types:  
        IIDTypeFiles=IIDFiles[IIDFiles.str.slice(0,len(Type))==Type]

        futures=[]
        with ProcessPoolExecutor(numCores) as executor: 
            for IIDTypeFile in IIDTypeFiles:
                futures+=[executor.submit(DBRead,name+'iid/'+IIDTypeFile,parms,True)]

            for f in wait(futures,return_when=ALL_COMPLETED)[0]:
                iid=f.result()
                iidStats[Type]+=[iid]

        print('loaded '+Type,flush=True)

    for Type in Types:
        iidStats[Type]=np.sort(np.concatenate(iidStats[Type],axis=0))
        DBLog(Type+' len(iidStats) '+str(len(iidStats[Type])),parms)
                        
    futures=[]        
    with ProcessPoolExecutor(numCores) as executor: 
        for Type in Types:  
            for file in statFiles[statFiles.str.contains(Type)]:
                #makePValsHelp(file,iidStats[Type],parms)
                futures+=[executor.submit(makePValsHelp,file,iidStats[Type],parms)]

        wait(futures,return_when=ALL_COMPLETED)
        
    return()

def makePValsHelp(TypeFile,iid,parms):
    name=parms['name']

    stat=DBRead(name+'stats/'+TypeFile,parms,True)
    sortOrd=np.argsort(stat)
    firstOrd=np.argsort(sortOrd)
    pval=1-np.searchsorted(iid,stat[sortOrd])/(len(iid)+1)
    pval=pval[firstOrd]

    DBLog(TypeFile+' stat ['+str(len(stat))+','+str(min(stat))+
          ','+str(max(stat))+'] pval ['+str(len(pval))+','+str(min(pval))+','+str(max(pval))+']',parms)
    DBWrite(pval,name+'pvals/'+TypeFile,parms,True)

    print('finished '+TypeFile,flush=True)
    
    return()
