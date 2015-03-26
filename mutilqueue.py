#!/usr/bin/python
from multiprocessing import Process,Queue
import sys,os

def f1(q):
#    q = Queue()
    while q.empty() == False :
        os.system(q.get(1)) 
        

def multi() :
    '''
    #args ./shell worklist.txt ProcessNum 
    #import time 
    #a =  time.ctime()
    #print 'multi start at ',a
    '''

    if len(sys.argv) < 4 :
        print 'argments number are invalid!'
        exit(1)
    f = open(sys.argv[1]+'/'+sys.argv[2],'r')
    commands = [ x.replace('\n','') for x in f.readlines() ]
    f.close()

    q = Queue()
   
    map(lambda x : q.put(x),commands)
    nprocess = int(sys.argv[3])
    p = [ Process(target=f1,args=(q,)) for x in range(0,nprocess)]
    map(lambda i : p[i].start(),range(0,nprocess))

    for i in range(0,nprocess):
        p[i].join()

if __name__=='__main__':
    multi()
