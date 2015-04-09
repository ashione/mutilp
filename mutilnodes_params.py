#!/usr/bin/python
import os,sys

if len(sys.argv)<2 :
    print 'please input commands file'
    exit(1)

comdfile = sys.argv[1]
f = open('slaves','r')
info = [x[:-1].split(' ')[0:3] for x in f.readlines()]
#print info,type(info)
nodes,cores,freq = [],[],[]

for item in info :
    nodes.append(item[0])
    cores.append(int(item[1]))
    freq.append(int(item[2]))

f.close()
cf = open(comdfile,'r')
comdata = [ x[:-1] for x in cf.readlines()]
cf.close()
print 'Now remove all nodes files'

map(lambda node : os.system("rm "+node),nodes)

capicitySum = sum([ x*y for (x,y) in zip(cores,freq) ] )

start = 0
dsum = 0
for i in range(0,len(nodes)):

    nf = open(nodes[i],'a+')
    end = start+ freq[i]*cores[i]*len(comdata)/capicitySum

    if i == len(nodes)-1:
        end = len(comdata)

    map(lambda x: nf.write(x+'\n'),comdata[start:end])
    nf.close()

    print i,len(comdata[start:end])
    dsum = dsum + end-start+1
    start = end

print 'commands total ',len(comdata),' converted commands ',start,dsum

print 'finished!'
