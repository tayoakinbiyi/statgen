import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from python.myStats import *
from python.ghc import *
from python.ggnull import *
from python.gbj import *

def fitStat(z,ebb,var,parms):
    power=pd.DataFrame()
    fail=pd.DataFrame()
        
    powerGG,failGG=ggnull(z,ebb,parms)
    power=power.append(powerGG)
    fail=fail.append(failGG)    
    print('ggnull',psutil.virtual_memory().percent)

    powerGHC,failGHC=ghc(z,var,parms)
    power=power.append(powerGHC)
    fail=fail.append(failGHC)    
    print('ghc',psutil.virtual_memory().percent)

    powerGBJ,failGBJ=gbj(z,parms)
    power=power.append(powerGBJ)
    fail=fail.append(failGBJ)    
    print('gbj',psutil.virtual_memory().percent)
    
    power=power.append(myStats(z,parms))
    print('myStats',psutil.virtual_memory().percent)
    
    return(power,fail)
