#!/usr/bin/python
import argparse,os
import numpy as np
import re
import collections
import scipy.stats

def readlines(filename):
    f = open(filename)
    rawdata = f.readlines()
    f.close()
    return rawdata
def cleanLineorR(line):
    p = re.compile(r'\r|\n')
    return p.sub('',line)
def generateScoreDictByRawData(filename):
    srf = readlines(filename)
    #print srf
    srt = re.compile(r'(\w+.\w+) img(\d{1,3}).bmp (\d+.\d+)')
    rawdict = map(lambda x: srt.sub(r'\2 \3',cleanLineorR(x)),srf)
    sdict = { int(xt[0]) :float(xt[1]) for xt in  map(lambda x: x.split(' '),rawdict) }
    return sdict
def genereteSocreByData(filename):
    srf = readlines(filename) #print srf #rawdict = map(lambda x: cleanLineorR(x),srf)
    #sdict = { xt[0] :float(xt[1]) for xt in  map(lambda x: x.split(' '),rawdict) }
    sdict = {}
    for i in range(0,len(srf)):
        sdict[(i)] = float(srf[i])
    return sdict
def getac(zipdata):
    m = 0
    n = 0
    for item in zipdata :
#        print item
        n +=1
        if str(item[0]) == str(item[1]):
            m+=1.0
    #print m,n
    return m/n
def getListFromDict(adict,classNum):
    alist = []
    for i in range(0,classNum):
        if type(adict[i]) is not tuple:
            alist.append(adict[i])
        else:
            if not adict[i][0] :
                print i,adict[i]
                raise ZeroDivisionError

            alist.append(adict[i][1]/adict[i][0])
    return alist
def argmaxFromSoftmax(ps):
    data = map(lambda x: map(lambda t : float(t),filter(lambda u :u!='',x.replace('\n','').split(' '))),readlines(ps.s)[2::3])
    data = np.array(data)
    data.shape = (data.size/ps.n,ps.n)
    return map(lambda x: np.argmax(x) ,data)

def argmaxFromSVM(ps):
    return map(lambda x : int(cleanLineorR(x)),readlines(ps.s))
#print data.shape

#print readlines(ps.l)[0]

p = argparse.ArgumentParser('iqa_softmax_score',usage='this is a script about counting score with iqa_net')
p.add_argument('-s',help="source outfile")
p.add_argument('-l',help="label file")
p.add_argument('-t',help="score file")
p.add_argument('-n',type=int,help="class num")
p.add_argument('-c', action="store_true", default=False)
p.add_argument('-d', help="store file", default="iqa_sorce.csv")

p.add_argument('-q', action="store_true", default=False)
ps = p.parse_args()

#print ps.s

classNum = ps.n 

label = map(lambda x : int(x.replace('\n','').split(' ')[-1]),readlines(ps.l))
#print label[0]
# argmax plus one because of starting from 0
argmax = argmaxFromSoftmax(ps) if not ps.q else argmaxFromSVM(ps)

pos =  zip(argmax,label)

if ps.c :
    sdict = generateScoreDictByRawData(ps.t)
else:
    sdict = genereteSocreByData(ps.t)

for k,v in sdict.items():
    if v == 0:
        sdict[k]= 100.0

#print sdict.items()

rdict = collections.defaultdict(lambda :(0,0.0) )

rdict_inner = collections.defaultdict(lambda :(0,0.0) )
#sdict[0] = 0.0

for x in pos :
   # print x
    rdict[int(x[1])]= (rdict[int(x[1])][0]+1,rdict[int(x[1])][1]+sdict[int(x[0])]) 

slist = getListFromDict(sdict,classNum)
if not ps.q :
    for i,x     in enumerate(pos) :
       # print x
        arg_sum = np.sum(np.dot(data[i].ravel(),slist))

        #rdict[int(x[1])]= (rdict[int(x[1])][0]+1,rdict[int(x[1])][1]+sdict[int(x[0])]) 
        rdict_inner[int(x[1])]= (rdict_inner[int(x[1])][0]+1,rdict_inner[int(x[1])][1]+arg_sum) 

fresult =  [(k, v[1]/v[0] )for k,v in rdict.items() ]
error = 0.0

#for i in range(1,classNum+1):
#for i in range(0,classNum):
    #print sdict[(i)],rdict[i][1]/rdict[i][0]
#    error+=(sdict[(i)]-rdict[i][1]/rdict[i][0]) ** 2

#print (error/classNum )** 0.5

print getac(pos)

#for pd,gt in zip(argmax,label):
#    print pd,gt

#print len(argmax),len(label)
#print 'spearmanr : ',scipy.stats.spearmanr(argmax,label)
#slist = getListFromDict(sdict,classNum)
rlist = getListFromDict(rdict,classNum)

if not ps.q :
    rlist_inner = getListFromDict(rdict_inner,classNum)
#print len(slist),len(rlist)
#print zip(slist,rlist)
#print 'spearmanr : ',scipy.stats.spearmanr(np.asarray(slist,dtype=int).ravel(),np.asarray(rlist,dtype=int).ravel())
print 'spearmanr : ',scipy.stats.spearmanr(slist,rlist)
print 'plcc:', scipy.stats.pearsonr(slist,rlist)

if not ps.q :
    print 'spearmanr(inner) : ',scipy.stats.spearmanr(slist,rlist_inner)
    print 'plcc(inner):', scipy.stats.pearsonr(slist,rlist_inner)

dumpdata = np.zeros((classNum,2),dtype=float)
dumpdata[:,0] = np.asarray(slist)
dumpdata[:,1] = np.asarray(rlist)
#print dumpdata
np.savetxt(ps.d, dumpdata, delimiter=",")
#print srf
#print max(argmax)
