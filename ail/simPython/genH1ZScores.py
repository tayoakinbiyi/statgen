from ail.opPython.DB import *
import numpy as np

# 4)
def genH1ZScores(parms):
    local=parms['local']
    name=parms['name']
    muList=parms['muList']
    epsList=parms['epsList']
    
    H1SnpSet=DBRead(name+'process/H1SnpSet.txt',parms,toPickle=False)
    traitData=DBRead(name+'process/traitData',parms,toPickle=True)

    path=local+name

    NullY=DBRead(name+'process/NullY',parms,toPickle=True)
    
    for mu in muList:
        for epsilon in epsList:
            if DBIsFile(name+'H1','z-'+str(mu)+'-'+str(epsilon),parms):
                continue
                
            DBWrite(np.array([]),name+'H1/z-'+str(mu)+'-'+str(epsilon),parms)
            
            futures=[]
            z=np.empty([H1SnpSet.shape[0],len(traitData)])

            with ProcessPoolExecutor(parms['cpu']) as executor: 
                for snp in range(H1SnpSet.shape[0]):
                    snpVec=H1SnpSet.iloc[snp,:].values.flatten()
                    futures.append(executor.submit(genH1ZScoreHelp,mu,epsilon,parms,snp,snpVec,NullY,len(traitData)))

                count=0
                for f in as_completed(futures):
                    z[count,:]=f.result()
                    count+=1

            DBWrite(z,name+'H1/z-'+str(mu)+'-'+str(epsilon),parms,toPickle=True)
            
    return()

def genH1ZScoreHelp(mu,epsilon,parms,snp,snpVec,NullY,numTraits):              
    f=np.avg(snpVec)/2
    A=np.zeros([1,numTraits])
    # add random effect size sign
    loc=np.random.choice(numTraits,size=epsilon,replace=False)
    pos=np.random.choice(loc,size=int(epsilon/2),replace=False)
    neg=np.array(set(loc)-set(pos))
    
    A[:,pos]=1
    A[:,neg]=-1
    Y=NullY+ (mu/np.sqrt(2*epsilon*f*(1-f)))*np.matmul(snpVec.reshape(-1,1),A)

    snpDF=pd.DataFrame([[range(len(snpVec)),'G','T']+snpVec.tolist()],index=[0]).to_csv(
        path+'H1/H1Snp-'+snp+'.txt',delimiter='\t',index=False,header=False)
    np.savetxt(local+name+'process/Y.txt',Y,delimiter='\t')
    
    ans=np.empty([1,numTraits])

    for trait in range(numTraits):
        cmd=['./gemma','-g',path+'H1/H1Snp-'+snp+'.txt','-p',path+'process/Y.txt','-lmm',pval,'-o',
             name[:-1]+'-'+str(snp)+'-'+str(trait+1),'-k',path+'process/grm-all.txt','-n',str(trait+1),'-c',path+'process/dummy.txt']

        subprocess.run(cmd) 

        df=pd.read_csv('output/H1-'+name[:-1]+'-'+str(snp)+'-'+str(trait)+'.assoc.txt',sep='\t')
        os.remove('output/H1-'+name[:-1]+'-'+str(snp)+'-'+str(trait)+'.assoc.txt')

        if wald:
            ans[trait]=(df['beta']/df['se']).iloc[0]
        else:
            ans[trait]=df['p_lrt'].iloc[0]
    
    return(ans)
