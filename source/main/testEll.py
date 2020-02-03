import pdb
import numpy as np
import sys
sys.path[0]=sys.path[0][:-5]

from scipy.stats import norm
import warnings

import ELL.ell as ell

N=2000
L=np.eye(N)
offDiag=np.matmul(L,L.T)[np.triu_indices(N)]
stat=ell.ell(offDiag,N,np.array([0.1,0.5])*N,reportMem=True)
# initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta
stat.fit(N*10,N*300,600,7,7)
M=200
x=stat.score(np.matmul(norm.rvs(size=[M,N]),L.T))
pMC=stat.monteCarlo(1e5,L)
pMarkov=stat.markov()
pdb.set_trace()
