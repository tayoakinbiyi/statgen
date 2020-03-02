from ail.opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb

def makeSimPVals(parms):
    name=parms['name']
    Types=parms['Types']
    numCores=parms['numCores']
    numSimStatSegments=parms['numSimStatSegments']
    
    DBCreateFolder(name,'pvals',parms)
    
    statFiles=pd.Series(DBListFolder(name+'stats',parms),name='statFiles')
    nameParms=['mu:'+str(x[0])+'-eps:'+str(x[1]) for x in parms['muEpsRange']]
                        
    for Type in Types:  
        for nameParm in nameParms:
            iid=np.concatenate([DBRead(name+'stats/'+Type+'-chr0-'+nameParm+'-'+str(segment),parms,True) for 
                segment in range(numSimStatSegments)])
            futures=[]        
            with ProcessPoolExecutor(numCores) as executor: 
                for segment in range(numSimStatSegments):
                    if DBIsFile(name+'holds','pvals-'+Type+'-'+nameParm,parms):
                        continue

                    DBWrite(np.array([]),name+'holds/pvals-'+Type+'-'+nameParm,parms,True)
                    
                    futures+=[executor.submit(makeSimPValsHelp,Type,nameParm,segment,iid,parms)]

                wait(futures,return_when=ALL_COMPLETED)
        
    return()

def makeSimPValsHelp(Type,nameParm,segment,iid,parms):
    name=parms['name']

    stat=np.concatenate([DBRead(name+'stats/'+Type+'-chr1-'+nameParm+'-'+str(segment),parms,True) for 
        segment in range(numSimStatSegments)])
    
    sortOrd=np.argsort(stat)
    firstOrd=np.argsort(sortOrd)
    
    pvals=1-np.searchsorted(iid,stat[sortOrd])/(len(iid)+1)
    pvals=pvals[firstOrd]
    
    DBLog(Type+'-'+nameParm+' len(stat) '+str(len(stat))+' len(pvals) '+str(len(pvals))+' minP '+str(min(pvals))+
          ' maxP '+str(max(pvals)),parms)
    DBWrite(pvals,name+'pvals/'+Type+'-'+nameParm+'-'+str(segment),parms,True)

    print('finished '+Type+'-'+nameParm,flush=True)
    return()
