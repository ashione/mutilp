#!/usr/bin/python
import os,sys
from ImageWalk import *

if __name__ == '__main__' :
    if len(sys.argv) < 3 :
        sys.exit(1)
    inputdir = sys.argv[1]
    outputfile = sys.argv[2]

    iwalk = ImageWalk(inputdir)
    imglist = iwalk.getImageFile()
    import re
    p = re.compile(r'\d{4}')
    f = open(outputfile,'w')
    for item in imglist :
        f.write(item + ' '+p.findall(item)[-1]+'\n')

    f.close()
    print 'Done!'

