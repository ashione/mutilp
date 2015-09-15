#!/usr/bin/python
import argparse,os
import numpy as np
import re
import collections
import scipy.stats
from scipy.optimize import curve_fit
import glog
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
                #raise ZeroDivisionError
                alist.append(0)
            else :
				alist.append(adict[i][1]/adict[i][0])
    return alist

def argmaxFromSoftmax(ps):
    data = map(lambda x: map(lambda t : float(t),filter(lambda u :u!='',x.replace('\n','').split(' '))),readlines(ps.s)[2::3])
    glog.info('Data Length : '+str(len(data)))
    data = np.array(data)
    data.shape = (data.size/ps.n,ps.n)
    return map(lambda x: np.argmax(x) ,data),data

def argmaxFromSVM(ps):
    return map(lambda x : int(cleanLineorR(x)),readlines(ps.s))

def logistic(X,beta1,beta2,beta3,beta4,beta5):
    logisticPart = 0.5 - 1./(1.0+np.exp(beta2*(X-beta3)))
    return beta1*logisticPart*beta4+beta5

def plcc(X,Y):
    try :
        popt,pconv = curve_fit(logistic,xdata=X,ydata=Y,p0=(np.max(Y),0,np.mean(X),0.1,0.1),maxfev=20000)
        #print 'popt: ',popt
        #print 'conv : ',pconv
    except RuntimeError,e:
        glog.info('RuntimeError {0}'.format(e))
    yhat = logistic(X,popt[0],popt[1],popt[2],popt[3],popt[4])
    #print zip(yhat,Y)
    glog.info('plcc: {0}'.format( scipy.stats.pearsonr(Y,yhat)))
    glog.info('rmse: {0}'.format( sum(((Y-yhat)**2)/yhat.size)**0.5))

def iqa_soft_score(ps):
    glog.info('Now couting:'+ps.s)

    classNum = ps.n 

    label = map(lambda x : int(x.replace('\n','').split(' ')[-1]),readlines(ps.l))
#print label[0]
# argmax plus one because of starting from 0
    if not ps.q :
        argmax,data = argmaxFromSoftmax(ps) 
    else :
        argmax =  argmaxFromSVM(ps)
    pos =  zip(argmax,label)

    if ps.c :
        sdict = generateScoreDictByRawData(ps.t)
    else:
        sdict = genereteSocreByData(ps.t)

    for k,v in sdict.items():
        if v == 0:
            sdict[k]= 0.0

#print sdict.items()

    rdict = collections.defaultdict(lambda :(0,0.0) )

    rdict_inner = collections.defaultdict(lambda :(0,0.0) )
#sdict[0] = 0.0

    for x in pos :
       # print x
        rdict[int(x[1])]= (rdict[int(x[1])][0]+1,rdict[int(x[1])][1]+sdict[int(x[0])]) 
#print sdict.keys()
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
    for i in range(0,classNum):
       #print sdict[(i)],rdict[i][1]/rdict[i][0]
        try :	
            error+=(sdict[(i)]-rdict[i][1]/rdict[i][0]) ** 2
        except Exception,e:
            glog.info('no type {0}'.format(i))

    glog.info('classNum : %d ; Error : %f ; Per class :%f ' % (classNum,error,(error /classNum)**0.5))

    glog.info('Testing Ac : '+str(getac(pos)))

#for pd,gt in zip(argmax,label):
#    print pd,gt

#print len(argmax),len(label)
    print 'spearmanr : ',scipy.stats.spearmanr(argmax,label)
    slist = getListFromDict(sdict,classNum)
    rlist = getListFromDict(rdict,classNum)

    if not ps.q :
        rlist_inner = getListFromDict(rdict_inner,classNum)
#print len(slist),len(rlist)
#print zip(slist,rlist)
#print 'spearmanr : ',scipy.stats.spearmanr(np.asarray(slist,dtype=int).ravel(),np.asarray(rlist,dtype=int).ravel())
    glog.info('spearmanr : {0}'.format(scipy.stats.spearmanr(slist,rlist)))
    glog.info('plcc[no nonfit]: {0}'.format( scipy.stats.pearsonr(slist,rlist)))
    glog.info('kenall : {0}'.format(scipy.stats.kendalltau(slist,rlist)))
#glog.info('*'*20)
    plcc(slist,rlist)
#glog.info('*'*20)
#plcc(rlist,slist)
    if not ps.q :
        glog.info( 'spearmanr(inner) : {0}'.format(scipy.stats.spearmanr(slist,rlist_inner)))
#    print 'plcc(inner):', scipy.stats.pearsonr(slist,rlist_inner)

    dumpdata = np.zeros((classNum,2),dtype=float)
    dumpdata[:,0] = np.asarray(slist)
    dumpdata[:,1] = np.asarray(rlist)
#print dumpdata
    np.savetxt(ps.d, dumpdata, delimiter=",")
#print srf
#print max(argmax)

def iqa_soft_type(ps):
    glog.info('Now couting:'+ps.s)

    classNum = ps.n 

    label = map(lambda x : int(x.replace('\n','').split(' ')[-1]),readlines(ps.l))
    imagesName = map(lambda x : '_'.join(x.replace('\n','').split(' ')[-2].split('/')[-1].split('_')[:-1]),readlines(ps.l))
#print label[0]
# argmax plus one because of starting from 0
    if not ps.q :
        argmax,data = argmaxFromSoftmax(ps) 
    else :
        argmax =  argmaxFromSVM(ps)
    pos =  zip(argmax,label)

    if ps.c :
        sdict = generateScoreDictByRawData(ps.t)
    else:
        sdict = genereteSocreByData(ps.t)

    for k,v in sdict.items():
        if v == 0:
            sdict[k]= 0.0

#print sdict.items()

    rdict = collections.defaultdict(lambda :(0,0.0) )

    rdict_inner = collections.defaultdict(lambda :(0,0.0) )
#sdict[0] = 0.0

    for i,x in enumerate(pos ):
       # print x
        rdict[imagesName[i]]= (rdict[imagesName[i]][0]+1,rdict[imagesName[i]][1]+sdict[int(x[0])]) 
    imagesSet = set(imagesName)
    rlist = []
    for x in imagesSet :
        rlist.append((x,rdict[x][1]/rdict[x][0]))
    fp = open(ps.d,'w') 
    map(lambda x : fp.write('%s %.5f\n'%(x[0],x[1])),rlist)
    fp.close()
        

#print max(argmax)
if __name__ == '__main__' :
    p = argparse.ArgumentParser('iqa_softmax_score',usage='this is a script about counting score with iqa_net')
    p.add_argument('-s',help="source outfile")
    p.add_argument('-l',help="label file")
    p.add_argument('-t',help="score file")
    p.add_argument('-n',type=int,help="class num")
    p.add_argument('-c', action="store_true", default=False)
    p.add_argument('-d', help="store file", default="iqa_sorce.csv")

    p.add_argument('-q', action="store_true", default=False) 
    p.add_argument('-type',action="store_true",default=False)
    ps = p.parse_args() 
    if not ps.type :
        iqa_soft_score(ps)
    else :
        iqa_soft_type(ps)
