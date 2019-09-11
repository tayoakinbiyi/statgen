from ail.opPython.DB import *
import numpy as np

# 4)
def genH1ZScores(parms):
    local=parms['local']
    name=parms['name']
    
    H1SnpSet=DBRead(name+'process/H1SnpSet.txt',parms,toPickle=False)
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)

    path=local+name

    H0Y=DBRead(name+'process/H0Y',parms,toPickle=True)
    
    futures=[]
    z=np.empty([H1SnpSet.shape[0],len(traitData)])
    
    with ProcessPoolExecutor(parms['cpu']) as executor: 
        for snp in range(H1SnpSet.shape[0]):
            snpVec=H1SnpSet.iloc[snp,:].values.reshape(1,-1)
            futures.append(executor.submit(genH1ZScoreHelp,parms,snpVec,H0Y,len(traitData)))
         
        count=0
        for f in as_completed(futures):
            z[count,:]=f.result()
            count+=1
      
    DBWrite(Z,name+'score/z-'+str(mu)+'-'+str(epsilon),parms,toPickle=True)
    return()

def genH1ZScoreHelp(parms,snpVec,H0Y,numTraits):              
    epsilon=parms['epsilon']
    mu=parms['mu']

    f=np.avg(snpVec)/2
    A=np.zeros([len(traitData),1])
    A[:,np.random.choice(len(traitData),size=epsilon,replace=False)]=1
    Y=H0Y+ (mu/np.sqrt(2*epsilon*f*(1-f)))*np.matmul(snpVec,A)

    snpDF=pd.DataFrame([[range(len(snpVec)),'G','T']+snpVec.tolist()],index=[0]).to_csv(
        path+'process/H1Snp.txt',delimiter='\t',index=False,header=False)
    np.savetxt(local+name+'process/Y.txt',Y,delimiter='\t')
    
    ans=np.empty([1,numTraits])

    for trait in range(numTraits):
        cmd=['./gemma','-g',path+'process/H1Snp.txt','-p',path+'process/Y.txt','-lmm',pval,'-o',name[:-1]+'-'+str(snp)+'-'+str(trait),
             '-k',path+'process/H1Snp.txt','-n',trait,'-c',path+'process/dummy.txt']

        print(' '.join(cmd),flush=True)
        subprocess.run(cmd) 

        df=pd.read_csv('output/'+name[:-1]+'-'+str(snp)+'-'+str(trait)+'.assoc.txt',sep='\t')
        os.remove('output/'+name[:-1]+'-'+str(snp)+'-'+str(trait)+'.assoc.txt')

        if wald:
            ans[trait]=(df['beta']/df['se']).iloc[0]
        else:
            ans[trait]=df['p_lrt'].iloc[0]
    
    return(ans)
