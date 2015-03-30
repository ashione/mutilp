#!/usr/bin/python
from Distribution import Distribution
if __name__ == '__main__':
#    print 'No errors'    
    f=open('slaves','r') 
    info = [ node.replace('\n','').split(' ') for node in f.readlines()]
    nodes = []
    nprocess = [] 
    #print info

    for i in range(0,len(info)):
        node,process = info[i][0],info[i][1]
        nodes.append(node)
        nprocess.append(process)
    f.close()

    print nodes
    terminalPath = 'mutilprocessHW'
    jobs = nodes 
    driver = 'mutilqueue.py'
    print jobs
    dt = Distribution(driver = driver,jobs=jobs,nodes=nodes,terminalPath=terminalPath,nprocess = nprocess)
    dt.nohupdisbale()
    dt.cleanJobDriver()
    dt.distributionJobDriver()
    dt.run()
