#!/usr/bin/python
import os,sys
#dirlist = ['AWAN','blur','contrast','fnoise','JPEG','JPEG2000']
fileprefix = '../'
#dirlist = ['%02d'%x for x in range(1,25)]
#dirlist = map(lambda x : os.path.join(fileprefix,x),dirlist)
dirlist = ['distorted_images']

def generatepkl(sfile):
    gdict = {}
    gdict['tr']=fileprefix+'label/'+sfile+'_clean_crop_train.txt'
    gdict['te']=fileprefix+'label/'+sfile+'_clean_crop_test.txt'
    gdict['fd']=sfile+'_crop'
    gdict['train'] = 'imgpydata/iqa_'+sfile+'_train.pkl'
    gdict['test'] = 'imgpydata/iqa_'+sfile+'_test.pkl'
    comstr = 'python ./imgpydata.py -tr={tr} -te={te} -fd={fd} -train={train} -test={test}'
    for k in gdict.keys():
        comstr = comstr.replace('{'+k+'}',gdict[k])
    print comstr
    os.system(comstr)

def generatelmdb(sfile):
    gdict = {}
    gdict['tr']=fileprefix+'label/'+os.path.basename(sfile)+'_clean_crop_train.txt'
    gdict['te']=fileprefix+'label/'+os.path.basename(sfile)+'_clean_crop_test.txt'
    gdict['fd']=sfile+'_crop'
    gdict['train'] = sfile+'_clean_crop_train_lmdb'
    gdict['test'] = sfile+'_clean_crop_test_lmdb'
    comstr = 'python ./imgpydata.py -tr={tr} -te={te} -fd={fd} -train={train} -test={test} -lmdb'
    for k in gdict.keys():
        comstr = comstr.replace('{'+k+'}',gdict[k])
    print comstr
    os.system(comstr)

#map(generatepkl,dirlist)
map(generatelmdb,dirlist)
