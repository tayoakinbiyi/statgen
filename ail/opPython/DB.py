import pickle
import os
import dropbox
import sys
import pdb
import datetime
import time
import numpy as np

def DBCreateFolder(path,parms):
    dropbox.Dropbox(parms['dbToken']).files_create_folder('/'+path)
    return()

def DBRead(path,parms,toPickle):
    path=('/'+path if len(path)>0 else path)
    dbx=dropbox.Dropbox(parms['dbToken'])
    
    md, data=dbx.files_download(path)
    
    if toPickle:
        data=pickle.loads(data.content)
        return(data)
    else:
        return(data.content)
        
    
def DBWrite(data,path,parms,toPickle): 
    path=('/'+path if len(path)>0 else path)
    dbx=dropbox.Dropbox(parms['dbToken'])

    if toPickle:
        data=pickle.dumps(data)
    
    size = len(data)

    chunkSize = 50*1024 * 1024

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
        
    return()

def DBUpload(path,parms,toPickle):
    local=parms['local']
    
    with open(local+path, 'rb') as f:
        data = f.read()
    
    DBWrite(data,path,parms,toPickle)
    return()
        
def DBIsFile(path,file,parms):
    return(file in DBListFolder(path,parms))

def DBListFolder(path,parms):
    path=('/'+path if len(path)>0 else path)
    dbx=dropbox.Dropbox(parms['dbToken'])
    
    return([x.name for x in dbx.files_list_folder(path).entries])

def DBSyncLocal(path,parms):
    dbx=dropbox.Dropbox(parms['dbToken'])
    local=parms['local']
    
    if not os.path.exists(local+path):
        os.mkdir(local+path)
    
    for file in [x.name for x in dbx.files_list_folder('/'+path).entries]:
        if os.path.isfile(local+path+'/'+file):
            continue
        print('syncing /'+path+'/'+file,flush=True)
        dbx.files_download_to_file(local+path+'/'+file,'/'+path+'/'+file)
            
    return()

def DBLocalRead(path,parms):
    local=parms['local']
    
    with open(local+path,'rb') as f:
        data=pickle.loads(f.read())

    return(data)

def DBLocalWrite(data,path,parms,toPickle):
    local=parms['local']
    
    with open(local+path,'wb') as f:
        if toPickle:
            f.write(pickle.dumps(data))
        else:
            f.write(data)

    return()