import pickle
import os
import dropbox
import sys
import pdb
import datetime
import time
import numpy as np

def DBCreateFolder(path,parms):
    dropbox.Dropbox(parms['dbToken']).files_create_folder(path)
    return()
    
def DBWrite(data,path,parms,toPickle=True):   
    dbx=dropbox.Dropbox(parms['dbToken'])
    if path='/'+path

    if toPickle:
        data=pickle.dumps(data)
    
    size = len(data)

    chunkSize = 1024 * 1024

    try:
        if size <= chunkSize:
            dbx.files_upload(data, path, dropbox.files.WriteMode.overwrite,mute=False)
        else:
            upload_session_start_result = dbx.files_upload_session_start(data[:chunkSize])
            
            cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,offset=chunkSize)
            commit = dropbox.files.CommitInfo(path=path)

            while ((size - cursor.offset) < chunkSize):
                dbx.files_upload_session_append_v2(data[cursor.offset:cursor.offSet+chunkSize], cursor)
                cursor.offset += chunkSize
                
            dbx.files_upload_session_finish(data[cursor.offset:],cursor,commit)
    except dropbox.exceptions.ApiError as err:
        print('*** API error', err.user_message_text)
        sys.exit(1)
        
    return()

def DBUpload(path,parms,toPickle):
    local=parms['local']
    
    with open(local+path, 'rb') as f:
        data = f.read()
    
    DBWrite(data,path,parms,toPickle)
    return()
        
def DBIsFile(path,file,parms):
    dbx=dropbox.Dropbox(parms['dbToken'])
    isFile=0
    try:
        res=dbx.files_list_folder(path)
        repeat=True
        while repeat:
            repeat=res.has_more
            isFile+=np.sum([(1 if x.name==file else 0) for x in res.entries])
            if repeat:
                res=parms['dbx'].files_list_folder_continue(res.cursor)
    except dropbox.exceptions.ApiError as err:
        print(file,'error',flush=True)
        return(False)
    
    return(isFile>0)

def DBSyncLocal(path,parms):
    dbx=dropbox.Dropbox(parms['dbToken'])
    local=parms['local']
    
    if not os.path.exists(local+path):
        os.mkdir(local+path)
    
    for file in [x.name for x in dbx.files_list_folder(path).entries]:
        if os.path.isfile(local+path+'/'+file):
            continue
        print('syncing /'+path+'/'+file,flush=True)
        try:
            dbx.files_download_to_file(local+path+'/'+file,'/'+path+'/'+file)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err.user_message_text)
            sys.exit(1)
            
    return()

def DBLocalRead(file,parms):
    local=parms['local']
    
    with open(local+file,'rb') as f:
        data=pickle.loads(f.read())

    return(data)

def DBLocalWrite(data,file,parms):
    local=parms['local']
    
    with open(local+file,'wb') as f:
        f.write(pickle.dumps(data))

    return()