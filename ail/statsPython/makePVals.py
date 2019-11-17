from ail.opPython.DB import *
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED
import pdb

def makePVals(parms):
    name=parms['name']
    Types=parms['Types']
    numCores=parms['numCores']
    
    DBWipe('pvals',parms)
    
    for Type in Types:  
        ref=DBRead('ref/'+Type,parms)
        DBLog(Type+' len(ref) '+str(len(ref)),parms)

        print('loaded '+Type,flush=True)

        stat=DBRead('stats/'+Type,parms)
        stat.insert(0,'ind',range(len(stat)))
        stat=stat.sort_values(by='Value')
        stat.insert(0,'pval',1-ref.searchsorted(stat['Value'])/(len(ref)+1))
        stat=stat.sort_values(by='ind')

        DBLog(TypeFile+' stat ['+str(len(stat))+','+str(stat['Value'].min())+
              ','+str(stat['Value'].mean())+','+str(stat['Value'].max())+'] pval ['+str(len(pval))+
              ','+str(pval.min())+','+str(pval.mean())+','+str(pval.max())+']',parms)
        DBWrite(stat['Value'],'pvals/'+Type,parms)

        print('finished '+Type,flush=True)
        
    return()
