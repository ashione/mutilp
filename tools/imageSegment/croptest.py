from ImgCrop import ImgCrop 
from ImageWalk import ImageWalk
import os,sys
import commands

dirlist = ['fastfading','gblur','jp2k','jpeg','wn']

for mdir in dirlist :
    print mdir
    if os.path.exists(mdir+'_crop'):
        os.system('rm -rf '+mdir+'_crop 2>&1 > /dev/null')
    os.mkdir(mdir+'_crop')

    print 'remove successful!'

    iw = ImageWalk(mdir)
    l = iw.getImageFile()
    #print l
    tm = ImgCrop(l,outdir = mdir+'_crop')
    tm.crop()

    print 'crop successful!'
