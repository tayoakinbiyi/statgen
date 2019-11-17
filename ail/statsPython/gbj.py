import pandas as pd
import numpy as np
import pdb
import subprocess
from ail.opPython.DB import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

def gbj(z,pval,nameParm,parms,folder,offDiag): 
    name=parms['name']
    local=parms['local']
      
    gbj=importr('GBJ')
    
    N=z.shape[1]
    
    i_vec=np.arange(1,N+1)
    
    gbjStats=[]
    ghcStats=[]
    for row in range(len(pval)):
        z_vec=ro.FloatVector(tuple(z[row]))
        p_vec=ro.FloatVector(tuple(pval[row]))
        d=ro.FloatVector((N,))
        pairwise_cors=ro.FloatVector(tuple(offDiag))
        mu=ro.FloatVector((0,))
        
        gbjStats+=[max(gbj.GBJ_objective(t_vec=z_vec,d=d,pairwise_cors=ro.FloatVector(offDiag)))]
        
        ghcStats+=[max((i_vec - N*pval[row]) / np.sqrt(gbj.calc_var_nonzero_mu(d=d, mu=mu,t=z_vec,pairwise_cors=pairwise_cors)))]

    stats=pd.concat([pd.DataFrame({'Type':'gbj','Value':np.array(gbjStats)}),
        pd.DataFrame({'Type':'ghc','Value':np.array(ghcStats)})],axis=0)
    
    DBLog('gbj '+nameParm+' len:min:max '+str(sum(~np.isnan(gbjStats)))+' : '+str(min(gbjStats))+' : '+
             str(max(gbjStats))+'\nghc '+nameParm+' len:min:max '+str(sum(~np.isnan(ghcStats)))+' : '+str(min(ghcStats))+' : '+
             str(max(ghcStats)),parms)
        
    print('gbj '+nameParm,flush=True)

    return(stats)
        