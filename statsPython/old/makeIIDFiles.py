import numpy as np
from ail.opPython.DB import *
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

def makeIIDFiles(parms):    
    IIDReps=parms['IIDReps']
    name=parms['name']
    Types=parms['Types']
    numCores=parms['numCores']

    files=DBListFolder(name+'iid',parms)
    futures=[]
    with ProcessPoolExecutor(numCores) as executor: 
        for Type in Types: 
            TypeFiles=[x[0] for x in zip(files,[Type]*len(files)) if x[0][:len(x[1])]==x[1]]
            futures+=[executor.submit(makeIIDFilesHelp,Type,TypeFiles,parms)]
        
        wait(futures,return_when=ALL_COMPLETED)
    
    return()

def makeIIDFilesHelp(Type,TypeFiles,parms):
    name=parms['name']
    pvalCutOff=parms['pvalCutOff']

    print('starting makeIIDFiles : Type : '+Type,flush=True)
    stat=np.sort(np.concatenate([DBRead(name+'iid/'+x,parms,True) for x in TypeFiles],axis=0))                        
    DBWrite(stat,name+'sim/'+Type,parms,True)
    print('finished makeIIDFiles : Type : '+Type,flush=True)
    
    return()

