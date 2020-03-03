import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta, bernoulli
import psutil
import time

from python.ggof import *
from python.myStats import *
from python.ghc import *
from python.ggnull import *
from python.gbj import *
from python.fitStat import *

def monteCarlo(L,eps,mu,p,ebb,parms):
    N=parms['N']
    Reps=parms['reps']
    cpus=parms['cpus']
    sigma=parms['sigma']
    K=parms['K']
    H=parms['H']
    
    zeta,sigmaHat,z=genDataSet(sigma,K,N,epsilon,mu,p) 
    power,Type1,fail=findPower(ebb,H,N,sigmaHat,z,mu,p,epsilon,parms)
    
    return(power,Type1,fail)

