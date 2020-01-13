import ELL.ell as ell
import pdb
import numpy as np

N=500
stat=ell.ell(np.array([.3]),N,np.array([0.1,0.5])*N,reportMem=True)
stat.fit(N*70,N*500,1e-3,500)
