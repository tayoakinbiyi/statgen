import warnings
import matplotlib
matplotlib.use('agg')

from python.monteCarlo import *
from python.norm_sig import *
from python.genAlpha import *

import numpy as np
import pdb
import os  

home='/home/akinbiyi/ail/'
#home='/home/ubuntu/ail/'

files={
    'dataDir':home+'data/',
    'scratchDir':home+'scratch/',
    'gemma':home+'gemma'
}

#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']
traitInfo=pd.read_csv(scratchDir+'traitInfo.csv')    

sig=np.eye(300)
parms={
    'plot':True,
    'H0':20,
    'fontsize':17,
    'new':True,
    'path':home,
    'Rpath':home+'ggnull/R',
    'cpus':cpu_count(),
    'binPower':500,
    'eps':1e-10,
    'delta':.1,
    'H0':30,
    'dataDir':files['dataDir'],
    'scratchDir':files['scratchDir'],
    'sigName':'test',
    'snpChr':['chr'+str(x) for x in range(1,20)],
    'traitChr':['chr'+str(x) for x in range(1,21)],
    'N':len(sig)
}

# genZ(sig,K,N,epsilon,p,mu) 
# sig is true sigma of z scores across traits for a given snp
# K is number of snps in data set, eg 523,000
# N is number of traits, eg 16 000
# p is prob a snp is an eqtl for any trait
# epsilon is percentage of traits it is an eqtl for
###
# zeta_k iid Ber(p), k=1,...,K 
# theta_ki|zeta_k ~ iid mu*Ber(zeta_k*epsilon), k=1,...,K, i=1,...,N
# U_k \sim iid mvn_N(0, sig), k=1,...,K
# Z_k | theta_k, U_j, U_{j-1} = theta_k + sqrt(alpha)*U_{j-1} + sqrt(1-alpha)*U_j, k=1,...,K
# sigHat=\hat{V}(Z_1,...,Z_K)
# return {zeta_k: k=1,..,K}, sigHat, {Z_k: k=1,...,K}
###############################################################3

# genPval(sig,H,N,{Z_k: k=1,...,K})
# H is number of reps for null distribution
# 
###
# X = {gbj, ELL,...,GHC,...,FDR, minP}
# \hat{Z}_j ~ iid MVN_N(0,sig), j=1,...,H
# \hat{S}_k = (\hat{S}_kx = f_x(\hat{Z}_k,sig) : x in X), j=1,...,H
# S_k = S_k(Z_k) = (S_kx = f_x(Z_k,sigHat) : x in X) k=1,...,K
# P_k = P_k(S_k,{\hat{S}_1,...,\hat{S}_H}) = (P_kx = (1/H)*sum_{j=1}^{H} I[S_kx <= \hat{S}_kx)] : x in X), k=1,...,K
# power_x = (1/|{k : \zeta_k=1}|)*\sum_{k : \zeta_k=1}I[P_kx<=C], x in X
# Type1_x = (1/|{k : \zeta_k=0}|)*\sum_{k : \zeta_k=0}I[P_kx<=C], x in X
# return {power_x : x in X}, {Type1_x : x in X}
###############################################################3


# {zeta_k: k=1,..,K}, sigHat, {Z_k: k=1,...,K} = genZ(sig, 523000, 16000, epsilon, p, mu)
# {power_x : x in X}, {Type1_x : x in X} = genPval(sigHat, 50000, 16000, {Z_k: k=1,...,K})
# 
'''
for trait in snpChr:
    parms['sigName']='ailfit-'+trait

    sig=np.loadtxt(scratchDir+'corr.csv',delimiter=',')    
    sig=sig[traitInfo.chromosome!=parms['chr'],traitInfo.chromosome!=parms['chr']]
    
    alpha,alphaPct,fail=genAlpha(parms,sig)
    parms['N']=len(sig)
'''    

alpha,alphaPct,fail=genAlpha(parms,sig)            