#!/usr/bin/python
import argparse,os
import numpy as np
import re
import collections
def readlines(filename):
    f = open(filename)
    rawdata = f.readlines()
    f.close()
    return rawdata

p = argparse.ArgumentParser('iqa_softmax_score',usage='this is a script about counting score with iqa_net')
p.add_argument('-s',help="source outfile")
p.add_argument('-l',help="label file")
p.add_argument('-t',help="score file")
ps = p.parse_args()

#print ps.s

classNum = 174
data = map(lambda x: map(lambda t : float(t),x.replace('\n','').split(' ')),readlines(ps.s)[3::4])
data = np.array(data)
data.shape = (data.size/classNum,classNum)
print data.shape

label = map(lambda x : x.replace('\n','').split(' ')[-1],readlines(ps.l))
# argmax plus one because of starting from 0
argmax = map(lambda x: np.argmax(x)+1 ,data)

pos =  zip(argmax,label)

srf = readlines(ps.t)
srt = re.compile(r'(\w+.\w+) img(\d{1,3}).bmp (\d+.\d+)\r\n')
rawdict = map(lambda x: srt.sub(r'\2 \3',x),srf)

sdict = { xt[0] :float(xt[1]) for xt in  map(lambda x: x.split(' '),rawdict) }

for k,v in sdict.items():
    if v == 0:
        sdict[k]= 100.0

#print sdict.items()

rdict = collections.defaultdict(lambda :(0,0.0) )

for x in pos :
    print x
    rdict[int(x[1])]= (rdict[int(x[1])][0]+1,rdict[int(x[1])][1]+sdict[str(x[0])]) 

fresult =  [(k, v[1]/v[0] )for k,v in rdict.items() ]
error = 0.0

for i in range(1,classNum+1):
    print sdict[str(i)],rdict[i][1]/rdict[i][0]
    error+=(sdict[str(i)]-rdict[i][1]/rdict[i][0]) ** 2

print error

#print srf
#print max(argmax)
