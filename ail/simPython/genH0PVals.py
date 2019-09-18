from ail.opPython.DB import *
from ail.simPython.getPVals import *
import numpy as np

def genH0PVals(parms):
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    maxZReps=parms['maxZReps']
    name=parms['name']
    
    snpData=DBRead(name+'process/snpData',parms,toPickle=False)
    traitData=DBRead(name+'process/traitData',parms,toPickle=False)
    
    if DBIsFile(name+'sim','pvals-0-0',parms):
        return()
    
    for snp in snpChr:        
        z=np.empty([sum(snpData==snp),len(traitData)])
        
        for trait in traitChr:
            z[:,traitData==trait]=DBRead(name+'score/p-'+snp+'-'+trait,parms,toPickle=True)
            
        Reps=len(z)
        
        for i in range(int(np.ceil(Reps/np.ceil(Reps/maxZReps)))):
            if DBIsFile(name+'sim','Null-'+snp+'-'+str(i),parms):
                continue
            
            DBWrite(pd.DataFrame(),name+'sim/Null-'+snp+'-'+str(i),parms,toPickle=True)
            zSegment=z[i*int(np.ceil(Reps/maxZReps)):min((i+1)*int(np.ceil(Reps/maxZReps)),Reps)]
            lenSegment=zSegment.shape[0]
            DBWrite((lenSegment,getPVals(zSegment,0,0,parms)),name+'sim/Null-'+snp+'-'+str(i),parms,toPickle=True)
        
    pvals=pd.DataFrame()
    for file in DBListFolder(name+'sim',parms):
        if not (file[0:4]=='Null-'):
            continue
        
        pvals=pvals.append(DBRead(name+'sim/'+file,parms,toPickle=True))

    DBWrite(pvals,name+'sim/pvals-0-0',parms,toPickle=True)
    
    return()

