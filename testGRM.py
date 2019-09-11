import os
from ail.opPython.DB import *
import pdb

parms={
    'local':os.getcwd()+'/',
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo'
}
td=DBRead('natalia/process/traitData',parms)
natalia=DBRead('natalia/score/p-chr18-chr1',parms)
nataliaFullGRM=DBRead('nataliaFullGRM/score/p-chr18-chr1',parms)
xx=np.concatenate([natalia[:,19].reshape(-1,1),nataliaFullGRM[:,19].reshape(-1,1)],axis=1)
pdb.set_trace()