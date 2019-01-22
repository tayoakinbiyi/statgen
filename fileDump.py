import plotly.plotly as py
import plotly.graph_objs as go
import pdb
import numpy as np
import pandas as pd

def fileDump(parms):
    j=0
    rawStats=parms[j];j+=1
    N=parms[j];j+=1
    H0=parms[j];j+=1
    H1=parms[j];j+=1
    sigName=parms[j];j+=1
    
    nonFail=rawStats[~rawStats.Type.str.contains('fail')]
    
    alpha=nonFail[nonFail.nullHyp].groupby(['mu','eps','nullHyp','Type'],sort=False).apply(
        lambda df:np.nanpercentile(df.Value,q=95))
    alpha.name='alpha'
    alpha=alpha.reset_index()
    
    power=nonFail[~nonFail.nullHyp].merge(alpha.drop(columns=['nullHyp','mu','eps']),on=['Type']).groupby(
        ['mu','eps','nullHyp','Type']).apply(lambda df:np.nanmean(df.Value>=df.alpha))
    power.name='power'
    power=power.reset_index()
    power.drop(columns='nullHyp',inplace=True)
    
    power=power.groupby(['mu','eps'],sort=False).apply(lambda df:pd.concat([df,pd.Series(df.power.rank(ascending=False),
        name='r')],axis=1)).sort_values(by=['mu','eps','r'],ascending=[False,False,True],inplace=True)

    pdb.set_trace()
    power.groupby(['Type']).apply(lambda df:py.iplot([go.Heatmap(x=df.mu,y=df.eps,z=df.power)],filename=
        str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-'+df.Type.iloc[0]+'-power.png',title=str(N)+'-'+sigName+'-'+str(H0)+'-'+
        str(H1)+'-'+df.Type.iloc[0]+'-power'))
    power.groupby(['Type']).apply(lambda df:py.iplot([go.Heatmap(x=df.mu,y=df.eps,z=df.r)],filename=
        str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-'+df.Type.iloc[0]+'-rank.png',title=str(N)+'-'+sigName+'-'+str(H0)+'-'+
        str(H1)+'-'+df.Type.iloc[0]+'-rank'))
    power.to_csv(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-'+df.Type.iloc[0]+'-power.csv')
    pdb.set_trace()
    
    rawStats.to_csv(str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-raw.csv')
    
    fail=rawStats[rawStats.Type.str.contains('fail')]
    failAvg=fail.groupby(['mu','eps','nullHyp','Type']).apply(lambda df:pd.DataFrame({'avg':np.mean(df.Value),
        'all':np.mean(df.Value==1)},index=[0]))
    failAvg.reset_index(inplace=True)
    failAvg.groupby(['Type']).apply(lambda df:py.iplot([go.Heatmap(x=df.mu,y=df.eps,z=df.avg)],filename=
        str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-'+df.Type.iloc[0]+'-avgFail.png'))
    failAvg.groupby(['Type']).apply(lambda df:py.iplot([go.Heatmap(x=df.mu,y=df.eps,z=df.all)],filename=
        str(N)+'-'+sigName+'-'+str(H0)+'-'+str(H1)+'-'+df.Type.iloc[0]+'-pctAllFail.png'))