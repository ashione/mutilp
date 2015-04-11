#!/usr/bin/python
import os,sys
import numpy as np
import scipy as scyp
def trimLine(line):
    x = line.replace('\n','')
    return x.replace('\r','').split(' ')

def trimeSpace(line):
    return len(line)>0

def readDataFromFile(file):
    f = open(file,'r') 
    lines = f.readlines()
    f.close()
    print 'Read Data From ',file,'succeed!'
    return map(trimLine,lines)

def writeDataToFile(file,lines):
    f = open(file,'w') 

    for buckItem in lines :
        map(lambda x :f.writelines(x),buckItem)

    f.close()
    print 'Write Data To ',file,'succeed!'

def writeToFlow(data,filename,flag,headx='/home/ydj/ydj_data/HMDB51/broxXframe/',heady='/home/ydj/ydj_data/HMDB51/broxYframe/'):
    if flag :
        xbuck = []
        ybuck = [] 
        for it in data :
            xbuck.append(map(lambda x : headx+x+'X.jpg'+' '+x.split('_')[0]+'\n',it))
            ybuck.append(map(lambda y : heady+y+'Y.jpg'+' '+y.split('_')[0]+'\n',it))
        writeDataToFile(lines = xbuck,file = 'Xshuffle'+filename)
        writeDataToFile(lines = ybuck,file = 'Yshuffle'+filename)
    else :
        buck = []
        for it in data :
            buck.append(map(lambda x : x+'.jpg'+' '+x.split('_')[0]+'\n',it))
        writeDataToFile('Shuffle'+filename,buck)

if __name__ =='__main__' :
    '''
    input params : 
        0 scrpit filename
        1 input data file name ( format is (FrameTotalNum,VideoNum)
        2 l - length of continueous frame 
        3 output optical flow filename
        4 output optical flow file flag : whether divide two segment
    #    5 absolutely path 
    '''
    if len(sys.argv)<5: 
        sys.exit(1)

    inputFile = sys.argv[1]
    l = int(sys.argv[2])
    oxf = sys.argv[3]
    flag = sys.argv[4]
    #abpath = sys.argv[5]

    data = readDataFromFile(inputFile)

    sumCtnFrame = 0
    buck = []
    for item in data :
        #print item
        sumCtnFrame = sumCtnFrame + int(item[0])

        #print [i for i in range(0,int(item[0]),l)]
        for i in range(0,int(item[0])-1,l):
            start = i
            end = (i+l)
            if start >= int(item[0])-1 :
                break
            if end >= int(item[0])-1 :
                end = int(item[0])-1
            #if/home/ydj/ydj_data/HMDB51/broxXframe/2672_35X.jpg item[1] =='2672':
              #  print start,end,item[0]
            tempbuck = [ item[1]+'_'+str(x) for x in range(start,end)]
            #print tempbuck
            #print len(tempbuck)
            #print tempbuck
            buck.append(tempbuck)
    np.random.shuffle(buck)
    writeToFlow(buck,oxf,flag=='1')

    #print sumCtnFrame
    #print len(buck)
#    print buck
    #print buck
#    print data

