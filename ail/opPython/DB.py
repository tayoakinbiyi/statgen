import pickle
import os
import dropbox
import sys
import pdb
import datetime
import time

def DBWrite(data,path,parms,toPickle=True):   
    path=('/' if len(path)>0 else '')+path
    if toPickle:
        data=pickle.dumps(data)
    
    try:
        res = parms['dbx'].files_upload(data, path, dropbox.files.WriteMode.overwrite,mute=True)
    except dropbox.exceptions.ApiError as err:
        print('*** API error', err.user_message_text)
        sys.exit(1)
        
    return()

def DBUpload(file,parms,toPickle):
    local=parms['local']
    
    with open(local+file, 'rb') as f:
        data = f.read()
    
    DBWrite(data,file,parms,toPickle)
    return()
        
def DBIsFile(folder,file,parms):
    path=('/' if len(path)>0 else '')+path
    isFile=0
    pdb.set_trace()
    try:
        res=parms['dbx'].files_list_folder(folder)
        repeat=False
        while repeat:
            repeat=res.has_more
            isFile+=np.sum([(1 if x.name==file else 0) for x in res.entries]
            if repeat:
                res=parms['dbx'].fildes_list_folder_continue(res.cursor)
    except dropbox.exceptions.ApiError as err:
        print(file,'error',flush=True)
        return(False)
    
    return(isFile>0)

def DBSyncLocal(folder,parms):
    local=parms['local']
    dbx=parms['dbx']
    
    if not os.path.exists(local+folder):
        os.mkdir(local+folder)
    
    for file in [x.name for x in dbx.files_list_folder('/'+folder).entries]:
        if os.path.isfile(local+folder+'/'+file):
            continue
        print('syncing /'+folder+'/'+file,flush=True)
        try:
            dbx.files_download_to_file(local+folder+'/'+file,'/'+folder+'/'+file)
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