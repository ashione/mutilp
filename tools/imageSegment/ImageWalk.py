#!/usr/bin/python
import os,sys

class ImageWalk :
    def __init__(self ,path='.'):
        self.path = path

    def isImage(self,filename):
        imgSuffix = ['jpg','bmp','png','gif']
        for suffix in imgSuffix :
            if os.path.splitext(filename)[1][1:].lower() == suffix :
                return True

        return False

    def getImageFile(self):
        imgList = []
        for rootName,dirName,fileName in os.walk(self.path):
            for fileN in fileName :
                if self.isImage(fileN):
                    imgList.append(os.path.join(rootName,fileN))

        return imgList

