#!/usr/bin/python
import os,sys
import re
from ImageWalk import ImageWalk

#dirlist = ['AWAN','blur','contrast','fnoise','JPEG','JPEG2000']
fileprefix = '../'
dirlist = ['%02d'%x for x in range(1,25)]
dirlist = map(lambda x : os.path.join(fileprefix,x),dirlist)
class DisType:
    def __init__(self,dirname=None):
        if dirname is None :
            raise NameError
        self.dirname = dirname

    def genlabeldict(self):
        with open(os.path.join(self.dirname,os.path.basename(self.dirname+'.txt'))) as fp:
            data = map(lambda line:line.split(' '),self.cleanspanline( fp.readlines() ))
            print data
            self.labeldict =  {  item[0] : (item[1],item[2]) for item in data}
            labelfile = open(fileprefix+'label/'+os.path.basename(self.dirname)+".txt",'w')
            map(lambda line : labelfile.write(line[2]+'\n'),data)
            labelfile.close()
            fp.close()

    def cleanspanline(self,lines):
        return map(lambda item : re.sub(r'\r|\n','',item),lines)

    def getDirImageList(self):
        iw = ImageWalk(self.dirname+'_crop')
        self.cropList = iw.getImageFile()

    def genrun(self):
        self.getDirImageList()
        self.genlabeldict()
        labelcols = map(lambda x: re.findall('(\w+\d+)_\d+(.\w+)',x),self.cropList)
        fp = open(fileprefix+'label/'+os.path.basename(self.dirname)+'_crop.txt','w')
        ftrain = open(fileprefix+'label/'+os.path.basename(self.dirname)+'_clean_crop_train.txt','w')
        ftest = open(fileprefix+'label/'+os.path.basename(self.dirname)+'_clean_crop_test.txt','w')
        for i,k in enumerate(self.cropList):
            info = (k.split('/')[-1]+' '+self.labeldict[''.join(labelcols[i][0])][0]+'\n')
            fp.write(info)
            if i%3 :
                ftrain.write(info)
            else :
                ftest.write(info)

        fp.close()
        ftrain.close()
        ftest.close()



for k in dirlist:
    temp = DisType(k)
    temp.genrun()

