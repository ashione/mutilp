#!/usr/bin/python
import numpy as np
import sys, lmdb, caffe,argparse,os
from caffe.proto import caffe_pb2  
sys.path.append("/home/tj/caffe/distribute/python/")



class TransLmdb(object):
    def __init__(self,lmdbfile=None,outputfile=None,labelfile=None):
        if lmdbfile is None or outputfile is None or labelfile is None :
            print 'lmdb or features file or lablefile input error'
            raise IOError
        if not os.path.exists(lmdbfile) and not os.path.exists(labelfile):
            raise RuntimeError

        self.lmdbfile = lmdbfile
        self.outputfile = outputfile
        self.labelfile = labelfile


    def run(self):
        output = open(self.outputfile, 'w') 
        db = lmdb.open(self.lmdbfile)
        txn = db.begin()
        it = txn.cursor()
        lfp = open(self.labelfile)
        labels = map(lambda x : x.replace('\r','').replace('\n','').split(' ')[-1] ,lfp.readlines())
        count = 0  
        lines ={} 
        for key,value in it:  
            datum = caffe_pb2.Datum.FromString(value)  
            arr = caffe.io.datum_to_array(datum).ravel()
            #print key  
            #output.write(str(labels[count]))
            #s.add(int(key))
            lines[int(key)] = str(labels[int(key)])
            for i,v in enumerate(arr):
                if np.abs(v) > 1e-6: 
                    #output.write(' '+str(i+1) + ':' + str(v) )
                    lines[int(key)] += ' '+str(i+1) + ':' + str(v) 

            #output.write('\n')  
            lines[int(key)]+='\n'
            count+=1  
            if not count%100 :
                print count  
        for i in range(len(lines.keys())):
            output.write(lines[i])

        output.close() 
        print 'Generate features file sucessful!'

if __name__ == '__main__':
    p = argparse.ArgumentParser('transfter lmdb to txt document',usage='this is a script about trainstform lmdb data to txt document')
    p.add_argument('lmdb',help='lmdb path')
    p.add_argument('ft',help='features filename')
    p.add_argument('label',help='label filename')
    ps = p.parse_args()
    tlmdb = TransLmdb(ps.lmdb,ps.ft,ps.label)
    tlmdb.run()
