#!/usr/bin/python
from Distribution import Distribution
if __name__ == '__main__':
#    print 'No errors'    
    f=open('slaves','r') 
    nodes = [ node.replace('\n','') for node in f.readlines()]
    f.close()

    print nodes
    terminalPath = 'mutilprocessHW'
    jobs = nodes 
    driver = 'mutilqueue.py'
    print jobs
    dt = Distribution(driver = driver,jobs=jobs,nodes=nodes,terminalPath=terminalPath)
    #dt.nohupdisbale()
    dt.cleanJobDriver()
    dt.distributionJobDriver()
    dt.run()
