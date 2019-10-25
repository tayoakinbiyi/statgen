suppressWarnings(suppressMessages(library(R.utils)))
library(GBJ)
library(reticulate)

args <- commandArgs(trailingOnly = TRUE)

j=1
nameParm=args[j];j=j+1
local=args[j];j=j+1
name=args[j];j=j+1
d=as.integer(args[j]);j=j+1
folder=args[j];j=j+1
offDiag=args[j];j=j+1
token=args[j];j=j+1
pyLoc=args[j];j=j+1
logName=args[j];j=j+1

path=paste(local,name,sep='')

import('numpy',convert=TRUE)
source_python(paste(pyLoc,'opPython/DB.py',sep=''),convert=TRUE)

z=read.csv(paste(path,'cpmaGBJ/z-',nameParm,'.csv',sep=''),header=FALSE)
pairwise_cors=as.matrix(read.csv(paste(path,'offDiag/offDiag-',offDiag,'.csv',sep=''),header=FALSE))

gbjArray=array(dim=c(dim(z)[1]))
ghcArray=array(dim=c(dim(z)[1]))

Reps=dim(z)[1]
for (row in 1:Reps) {
    pRow=as.numeric(z[row,])
    gbjArray[row]=max(GBJ_objective(t_vec=pRow, d=d, pairwise_cors=pairwise_cors))
}
for (row in 1:Reps) {
	i_vec <- 1:d
	p_values <- 1-pchisq(pRow^2, df=1)
    ghcArray[row]=max((i_vec - d*p_values) / sqrt(calc_var_nonzero_mu(d=d, t=pRow, mu=0,pairwise_cors=pairwise_cors)))
}

RDBLog(paste('gbj ',nameParms,'\nlen:min:max ',sum(!is.null(gbjArray)),' : ',min(gbjArray,na.rm=TRUE),' : ',
             max(gbjArray,na.rm=TRUE),'\nghc len:min:max ',sum(!is.null(ghcArray)),' : ',min(ghcArray,na.rm=TRUE),' : ',
             max(ghcArray,na.rm=TRUE),sep=''),token,logName)

RDBWrite(gbjArray,paste(name,folder,sep=''),token,TRUE): 
RDBWrite(ghcArray,paste(name,folder,sep=''),token,TRUE): 
