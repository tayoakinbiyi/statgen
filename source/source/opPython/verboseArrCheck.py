import numpy as np

def verboseArrCheck(arr):
    mean=str(np.mean(arr))
    mMin=str(np.min(arr))
    mMax=str(np.max(arr))
    mNA=str(sum(np.isnan(arr)))
    mLen=str(len(arr))
    
    return('min '+mMin+' mean '+mean+' max '+mMax+' NA '+mNA+' len '+mLen) 
                                  