#!/usr/bin/python
import re
from ImageWalk import ImageWalk

imw = ImageWalk('/home/tj/IQA_CNN/TID08/distorted_images_crop')
items = imw.getImageFile()
#re.match(r'.*i|I\d{2}_(\d+)_\d+_\d+.bmp')
itemgroups =  map(lambda x:re.match(r'.*[iI]\d+_(\d+)_\d+_\d+.bmp',x),items)
fp = open('/home/tj/IQA_CNN/TID08/label/all.txt','w')
for reitem in itemgroups :
    fp.write('{fname} {ftype}\n'.format(fname=reitem.group(),ftype=int( reitem.groups()[0] )-1))

fp.close()
