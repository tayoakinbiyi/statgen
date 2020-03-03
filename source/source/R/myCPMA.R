suppressWarnings(suppressMessages(library(R.utils)))
library(reticulate)

args <- commandArgs(trailingOnly = TRUE)
j=1
nameParm=args[j];j=j+1
local=args[j];j=j+1
name=args[j];j=j+1
folder=args[j];j=j+1
token=args[j];j=j+1
pyLoc=args[j];j=j+1
logName=args[j];j=j+1

path=paste(local,name,sep='')

source_python('ga/ail/opPython/DB.py',convert=TRUE)
sourceDirectory(paste(pyLoc,'R/cpma',sep=''))

pval=read.csv(paste(path,'cpmaGBJ/p-',nameParm,'.csv',sep=''),header=FALSE)

cpmaArray=array(dim=c(dim(pval)[1]))
d=dim(pval)[1]

for (row in 1:d) {
    cpmaArray[row]=cpma.score.exponential.nolog(as.numeric(pval[row,]))
}

RDBLog(paste('cpma ',nameParm,' len:min:max \t',sum(!is.null(cpmaArray)),' : ',min(cpmaArray,na.rm=TRUE),' : ',
     max(cpmaArray,na.rm=TRUE),sep=''),token,logName)

RDBWrite(cpmaArray,paste(name,folder,sep=''),token): 

