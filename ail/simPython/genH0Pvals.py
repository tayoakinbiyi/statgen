from ail.opPython.DB import *
import numpy as np

def genH0Pvals(parms):
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    maxZReps=parms['maxZReps']
    
    snpData=DBRead(name+'process/snpData',parms,toPickle=False)
    traitData=DBRead(name+'process/traitData',parms,toPickle=False)
    
    for snp in snpChr:        
        z=np.empty([sum(snpData==snp),len(traitData)])
        
        for trait in traitChr:
            z[:,traitData==trait]=DBRead(name+'score/z-'+snp+'-'+trait,parms,toPickle=True)
            
        Reps=len(z)
        
        for i in range(int(np.ceil(Reps/np.ceil(Reps/maxZReps)))):
            if DBIsFile(name+'sim/','H0P-'+snp+'-'+str(i),parms):
                continue
            
            DBWrite(np.array([]),name+'sim/H0P-'+snp+'-'+str(i),parms,toPickle=True)
            DBWrite(getPVals(z[i*int(np.ceil(Reps/maxZReps)):min((i+1)*int(np.ceil(Reps/maxZReps)),Reps)],parms),
                    name+'sim/H0P-'+snp+'-'+str(i),parms,toPickle=True)
            
    return()

