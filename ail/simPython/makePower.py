from ail.opPython.DB import *

import pandas as pd

def makePowerHelp(pvals,parms):
    name=parms['name']
    pvalCutOff=parms['pvalCutOff']

    N=DBRead(name+'process/N',parms,toPickle=False)
    
    power=pvals.copy()
    power.loc[:,'Value']=1*(power.loc[:,'Value']<pvalCutOff)
    power=(1000.0*power.groupby(['eps','mu','Type']).mean()).reset_index()
    
    return(power)

def makePower(pvals,parms):
    name=parms['name']
    
    if DBIsFile(name+'sim','power',parms):
        return()
    
    pvals=pd.DataFrame()
    for file in DBListFolder(name+'sim',parms):
        if not (file[0:4]=='pvals-'):
            continue

        pvals=pvals.append(DBRead(name+'sim/'+file,parms,toPickle=True))

    power=makePowerHelp(pvals)

    DBWrite(power,name+'sim/power',parms,toPickle=True)
        
    return()