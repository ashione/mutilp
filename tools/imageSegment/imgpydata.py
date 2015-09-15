import Image,os,sys
import time, argparse,re,cPickle
import numpy as np
fileprefix = '../'
def readlines(filename):
    f = open(filename)
    rawdata = [ cleanLineorR(x) for x in f.readlines()]
    f.close()
    return rawdata

def cleanLineorR(line):

    p = re.compile(r'\r|\n')
    return p.sub('',line)

def makeimgdata(flist,dct):
    start = time.time()
    data = {}
    fdata = map(lambda x : x.split(' '),flist)
    #print fdata[0:10]
    data['X'] = np.asanyarray([np.asarray(Image.open(os.path.join(dct,x[0])),dtype=np.uint8) for x in fdata],dtype=np.uint8)
    data['y'] = np.asarray([int(x[1]) for x in fdata],dtype=np.uint8)
    print 'generate data cost ',time.time()-start
    return data

def generatepkl(p):
    sflist = readlines(p.tr)
    nflist = readlines(p.te)
    #map(lambda (imga,imgb ): imgdifference(os.path.join(p.sd,imga),os.path.join(p.nd,imgb),p.od,imgb),zip(sflist,nflist))
    print len(sflist),len(nflist)
    #print sflist
    #data = makeimgdata(sflist,nflist,p.fd)
    traindata = makeimgdata(sflist,p.fd)
    testdata = makeimgdata(nflist,p.fd)

    trstart = time.time()
    f = open(p.train,'wb')
    cPickle.dump(traindata,f)
    f.close()
    print 'train data dump done!',time.time()-trstart

    testart = time.time()
    f = open(p.test,'wb')
    cPickle.dump(testdata,f)
    f.close()
    print 'test data dump done!',time.time()-testart

def testloadingpkl(dfile):

    start = time.time()
    data = cPickle.load(open(dfile,'rb'))
    print 'loading time cost was ',time.time()-start
    print data['y']

def generatelmdb(p):
    '''
    generate lmdb and meanfile
   '''
    import commands
    #cmd = ("~/caffe/distribute/bin/convert_imageset.bin "+p.fd+"/ "+p.train+' '+p.tr)
    #print cmd
    #print p.fd,p.tr,p.train
    commands.getstatusoutput("~/caffe/distribute/bin/convert_imageset.bin "+p.fd+"/ "+p.tr+' '+p.train)
    #print "~/caffe/distribute/bin/compute_image_mean.bin "+p.train+' '+fileprefix+'meanfile/'+'_'.join(str(os.path.basename(p.train)).split('_')[:-1])+'_mean.binaryproto'
    commands.getstatusoutput("~/caffe/distribute/bin/compute_image_mean.bin "+p.train+' '+fileprefix+'meanfile/'+'_'.join(str(os.path.basename(p.train)).split('_')[:-1])+'_mean.binaryproto')
    commands.getstatusoutput("~/caffe/distribute/bin/convert_imageset.bin "+p.fd+"/ "+p.te+' '+p.test)
    commands.getstatusoutput("~/caffe/distribute/bin/compute_image_mean.bin "+p.test+' '+fileprefix+'meanfile/'+'_'.join(str(os.path.basename(p.test)).split('_')[:-1])+'_mean.binaryproto')

ps = argparse.ArgumentParser('iqadata generation',usage='this is a script to generate iqa data')
ps.add_argument('-tr',help='the train image filename list',type=str,default='./label/fastfading_clean_crop_train.txt')
ps.add_argument('-te',help='the test image filename list',type=str,default='./label/fastfading_clean_crop_test.txt')
ps.add_argument('-fd',help='the original image filename direcotry',type=str,default='./fastfading_crop')
ps.add_argument('-train',help='the output train image data filename ',type=str,default='iqa_train.pkl')
ps.add_argument('-test',help='the output test image data filename ',type=str,default='iqa_test.pkl')
ps.add_argument('-lmdb',help="whether generate lmdb",action="store_true",default=False)
p = ps.parse_args()

if p.lmdb:
    generatelmdb(p)

#generatepkl(p)
#testloadingpkl(p.test)

