from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

def apply(df,cols, func):
    dfGrouped=df.groupby(cols)
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

