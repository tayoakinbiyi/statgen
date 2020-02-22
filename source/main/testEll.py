import pdb
import numpy as np
import sys
sys.path=['source']+sys.path

from scipy.stats import norm
import warnings

from ELL.ell import *

N=400
L=np.eye(N)
offDiag=np.array([0]*int(N*(N-1)/2))
stat=ell(np.array([.5]),offDiag)
stat.fit(1*N,300*N,1000,1e-6) # numLamSteps0,numLamSteps1,numEllSteps,minEll
# initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta
pdb.set_trace()
M=200
x=stat.score(np.matmul(norm.rvs(size=[M,N]),L.T))
pMC=stat.monteCarlo(1e5,L)
pMarkov=stat.markov()
pdb.set_trace()
