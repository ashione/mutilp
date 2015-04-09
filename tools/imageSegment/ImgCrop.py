#!/usr/bin/python
import os,sys
from ImageWalk import *
import Image

class ImgCrop :
    def __init__(self, imgfList, outdir,height = 32,width = 32) :
        
        self.imgfList = imgfList
        self.outdir = outdir 
        self.width,self.height = width,height

    def segmentImgeAndSuffix(self,filename):
        return os.path.splitext(os.path.split(filename)[1])

    def singleCrop(self,filename):
        im = Image.open(filename) 
        imH,imW = im.size
        num = 0
        for x in range(0,imH/self.height):
            for y in range(0,imW/self.width):
                cimg ,suffix = self.segmentImgeAndSuffix(filename)
                cimg = cimg+'_'+str(num)
                cimgfull = cimg+suffix
                im.crop((x*self.height,y*self.width,(x+1)*self.height,(y+1)*self.width)).save(os.path.join(self.outdir,cimgfull))
                num = num +1
        #print 'image ',filename,' OK'

    def crop(self):
        map(lambda fl : self.singleCrop(fl),self.imgfList)
        print 'Crop done!'

            
