#!/usr/bin/python
import os
import re,sh
import glog
import collections
fp = open('../mos_with_names.txt')
imgroot = '../distorted_images'
items =  map(lambda x: x.strip(),fp.read().strip().split(os.linesep))
print items
itemgroups =  map(lambda x:re.match('(\d.\d+) ((i\d{2})_(\d+)_\d+.bmp)',x),items)
itemgroups =  map(lambda f :  f.groups(),itemgroups)
numdict = collections.defaultdict(lambda : 0)
for item in itemgroups :
    categoryroot = os.path.join('..',item[3])
    labelroot = '../label/'
    if not os.path.exists(categoryroot):
        os.mkdir(categoryroot)
    glog.info('picture  : %s is coping' % item[1])
    print item
    towfilename = os.path.join(categoryroot,item[3]+'.txt')
    tolbfilename = os.path.join(labelroot,item[3]+'.txt')
    if numdict[item[3]] == 0 :
        if os.path.exists(towfilename):
            os.remove(towfilename)
    #os.popen('cp {0} {1}'.format(os.path.join(imgroot,item[1]),os.path.join(categoryroot,item[1])))

    with open(os.path.join(categoryroot,item[3]+'.txt'),'a') as p:
        p.write('%s %s %s\n' % (item[1],numdict[item[3]],item[0]))
        numdict[item[3]] += 1
        p.close()

    if numdict[item[3]] == 0 :
        if os.path.exists(tolbfilename):
            os.remove(tolbfilename)
    with open(tolbfilename,'a') as p:
        p.write('%s\n' % (item[0]))
    #    numdict[item[3]] += 1
        p.close()
