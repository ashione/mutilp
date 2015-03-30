#!/usr/bin/python
import os,sys

if len(sys.argv)<2 :
    print 'please input commands file'
    exit(1)

comdfile = sys.argv[1]
f = open('slaves','r')
nodes = [x[:-1].split(' ')[0] for x in f.readlines()]

f.close()
cf = open(comdfile,'r')
comdata = [ x[:-1] for x in cf.readlines()]
cf.close()
print 'Now remove all nodes files'
os.system("rm node*")

for i in range(0,len(comdata)):
    nf = open(nodes[i%(len(nodes))],'a+')
    nf.write(comdata[i]+'\n')
    nf.close()

print 'finished!'
