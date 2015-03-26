#/usr/bin/python
import os,sys

class Distribution :

    __nohup = True
    __nohupcommand = ' > /dev/null 2>&1 & '

    def __init__(self,terminalPath,nodes,driver,jobs,nprocess=8):
        '''
        terminalPath : terminalPath direcotry path
        driver : enodeecuable file 
        jobs : commands file
        nprocess : number of process when driver is running 
        '''
        self.terminalPath = terminalPath
        self.driver = driver
        self.nodes = nodes
        self.jobs = jobs
        self.nprocess = nprocess

    def nohupenbale(self) :
        self.__nohup = True

    def nohupdisbale(self):
        self.__nohup = False

    def distributionJobDriver(self):

        for (node,job) in zip(self.nodes,self.jobs) :
            if self.__nohup :
                os.system('ssh '+node+' " mkdir '+self.terminalPath+' '+self.__nohupcommand+'"')
            else :
                os.system('ssh '+node+' " mkdir '+self.terminalPath+' "')

            jd = ' '.join([self.driver,job])
#            print jd
            os.system('scp '+jd+' tj@'+node+':~/'+self.terminalPath+'/ ')
            #os.system('ssh ' +node+' "rm -rf '+terminalPath+'/*"')

        print 'push driver and jobs files into slaves finished!'

    def cleanJobDriver(self):
        for (node,job) in zip(self.nodes,self.jobs) :
            if self.__nohup :
                os.system('ssh '+node+' " rm -rf '+self.terminalPath+' '+self.__nohupcommand+'"')
            else :
                os.system('ssh '+node+' " rm -rf '+self.terminalPath+' "')

    def singlerun(self,node,job):

        jd = ' '.join([self.driver,self.terminalPath,job,str(self.nprocess)])

        if self.__nohup :
            fulljd = '/'.join([self.terminalPath,jd,self.__nohupcommand])
        else :
            fulljd = '/'.join([self.terminalPath,jd])

        os.system('ssh '+node+' "'+fulljd+' "')

    def run(self):
        map(lambda (u,v) : self.singlerun(u,v),zip(self.nodes,self.jobs))


