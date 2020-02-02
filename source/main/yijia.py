import dropbox
import numpy as np
import pdb
import sys
import pickle

sys.path[0]=sys.path[0][:-5]

token='YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo'

dbx=dropbox.Dropbox(token)

for chr in range(1,20):
    nm='p-chr'+str(chr)+'-chr1'
    md, data=dbx.files_download('/analysisChr1/score/'+nm)

    data=pickle.loads(data.content)
    
    np.savetxt('data/'+nm,data,delimiter=',')
    
    with open('data/'+nm,'rb') as f:
        data=f.read()
        
    size = len(data)

    chunkSize = 50*1024 * 1024
    path='/comparison/score/yijia-chr'+str(chr)+'-chr1.csv'

    if size <= chunkSize:
        dbx.files_upload(data, path, mode=dropbox.files.WriteMode.overwrite,mute=False)
    else:
        upload_session_start_result = dbx.files_upload_session_start(data[:chunkSize])

        cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,offset=chunkSize)
        commit = dropbox.files.CommitInfo(path=path,mode=dropbox.files.WriteMode.overwrite)

        while ((size - cursor.offset) > chunkSize):
            dbx.files_upload_session_append_v2(data[cursor.offset:cursor.offset+chunkSize], cursor)
            cursor.offset += chunkSize

        dbx.files_upload_session_finish(data[cursor.offset:],cursor,commit)
            
    
