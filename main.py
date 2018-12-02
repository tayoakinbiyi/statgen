import warnings
import numpy as np
from genL import *
from monteCarlo import *
#snap-05e3c2f3c7efd6df1
from mymath import *
#warnings.filterwarnings("error")

if __name__ == '__main__':
    
    N=1000
    delta=10
    H0=5000
    H1=200

    power=pd.DataFrame([])

    for sig in [[0,0,0]]:#[0.3,0,0.3],[0.3,0,0],
        for theta in np.linspace(.5+0.5/delta,1-0.5/delta,delta):
            for r in np.linspace(rho(theta)+(theta-rho(theta))/delta,theta-(theta-rho(theta))/delta,delta):
<<<<<<< HEAD
                parms+=[{'H0':H0,'H1':H1,'N':N,'mu':np.round(np.sqrt(2*r*np.log(N)),3),'eps':int(N**(1-theta)),
                     'rho1':sig[0],'rho2':sig[1],'rho3':sig[2],'r':np.round(r,3),'theta':np.round(theta,3)}]
                genL(parms[-1])

    try:
        pool = Pool(cpu_count())
        results=pool.map(monteCarlo, parms)
    finally:
        pool.close()
        pool.join()            

    power=pd.DataFrame([])
    for result in results:
        power=power.append(result)
=======
                parms={'H0':H0,'H1':H1,'N':N,'mu':np.round(np.sqrt(2*r*np.log(N)),3),'eps':int(N**(1-theta)),
                     'rho1':sig[0],'rho2':sig[1],'rho3':sig[2],'r':np.round(r,3),'theta':np.round(theta,3)}
                genL(parms)
                power=power.append(monteCarlo(parms))
>>>>>>> 0addef83998ee69bce10a16c4132192037a47e99

    power.to_csv('power.csv',index=False)