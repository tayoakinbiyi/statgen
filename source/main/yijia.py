import dropbox
import numpy as np
import pdb
import sys
import pickle

sys.path[0]=sys.path[0][:-5]

token='YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo'

pdb.set_trace()

dbx=dropbox.Dropbox(token)

for chr in range(1,20):
    nm='p-chr'+str(chr)+'-chr1'
    md, data=dbx.files_download('/comparison/score/'+nm)

    data=pickle.loads(data.content)
    np.savetxt('data/'+nm,data,delimiter='\t')
    
