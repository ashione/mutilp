# coding: UTF-8
import sys
import string
import math

def MatrixTransform(m,n,matrix):
    bank1=[]
    bank2=[]
    firm1=[]
    firm2=[]
    temp = []
    tempBank = []
    tempFirm = []
    maxBank=0.0
    maxFirm=0.0
    
    indexBank = []
    indexFirm = []
    
    for i in range(0, n):
        bank1.append([])
        bank2.append([])
        indexBank.append(0)
        for j in range(0, n): 
            bank1[i].append(0)
            bank2[i].append(0) 
    for i in range(0, m):
        firm1.append([])
        firm2.append([])
        indexFirm.append(0)
        for j in range(0, m):
            firm1[i].append(0)
            firm2[i].append(0)
    for i in range(0, m):
        print '\n',
        for j in range(0, n):
            print matrix[i][j],
   
   
    i=0   
    t=0
    while i!=m:
        tempFirm.append(0)
        for j in range(0, n):
            if indexFirm[i] == 0:
                indexFirm[i] = i
            tempFirm[i]=tempFirm[i]+matrix[i][j]
        if tempFirm[i] ==0:            
            del matrix[i];            
            t=t+1
            for k in range(i,m):                
                indexFirm[k] = k+t
            i=i-1
            m=m-1 
        i=i+1    
   
   
   
    i=0 
    t=0
    while i!=n:
        tempBank.append(0)
        for j in range(0, m):
            if indexBank[i] ==0:
                indexBank[i] = i
            tempBank[i]=tempBank[i]+matrix[j][i]    
        if tempBank[i] == 0:
            t=t+1
            
            for item in matrix:
                del item[i];    
            for k in range(i,n):                
                indexBank[k] = k+t     
            i = i-1       
            n=n-1   
        i=i+1
    
    print u'\n除去孤点###########################################\n'    
    for i in range(0, m):
        print '\n',
        for j in range(0, n):
            print matrix[i][j],        
    
    
    
    for i in range(0, m):
        del temp  
        temp = []        
        for j in range(0,n):
            temp.append(0)
            if(matrix[i][j]>0):
                temp[j]=1
        for j in range(0, n):
            for k in range(j+1, n):                
                if(temp[j]==1 and temp[k]==1):
                    bank1[j][k] = bank1[j][k]+1
                bank1[k][j] = bank1[j][k]
                if(maxBank<bank1[j][k]):
                    maxBank=bank1[j][k]
                    
                 
    del temp  
    temp = []
    for i in range(0, n):
        del temp  
        temp = []        
        for j in range(0,m):
            temp.append(0)
            if(matrix[j][i]>0):
                temp[j]=1
        for j in range(0, m):
            for k in range(j+1, m):                
                if(temp[j]==1 and temp[k]==1):
                    firm1[j][k] = firm1[j][k]+1
                firm1[k][j] = firm1[j][k]
                if(maxFirm<firm1[j][k]):
                    maxFirm=firm1[j][k]         
   
    nbank =n  
    i=0   
    t=0
    while i!=nbank:    
        flag=0
        for j in range(0, nbank):  
            if(bank1[i][j]!=0):
                flag=1
                break
        if flag==0:
            t=t+1
            for k in range(i,nbank):
                indexBank[k] = indexBank[k]+t  
            for iterm in bank1:
                del iterm[i]
            del bank1[i]                
            i=i-1    
            nbank = nbank-1
        i =i+1 
   
    mfirm =m  
    i=0     
    t=0
    while i!=mfirm:    
        flag=0
        for j in range(0, mfirm):  
            if(firm1[i][j]!=0):
                flag=1
                break
        if flag==0:
            t=t+1
   
            for k in range(i,mfirm):
                indexFirm[k] = indexFirm[k]+t
            for iterm in firm1:
                del iterm[i]
            del firm1[i]                 
            i=i-1    
            mfirm = mfirm-1
        i =i+1 
        
    for i in range(0, nbank):
        for j in range(0, nbank):
            bank1[i][j] = float(bank1[i][j])/maxBank
            
    for i in range(0, mfirm):
        for j in range(0, mfirm):
            firm1[i][j] = float(firm1[i][j])/maxFirm
            
    for i in range(0, n):
        flag =0
        for j in range(i+1 ,n):
            bank2[i][j]=0
            for k in range(0, m):
                bank2[i][j]=bank2[i][j]+((matrix[k][i])*(matrix[k][j]))/(tempBank[i]*tempBank[j])
                bank2[j][i]=bank2[i][j] 
#                if bank2[i][j]!=0:
#                    flag=1
#            if flag==0:
#                for iterm in bank2:
#                    del iterm[i]
#                del bank2[i] 
    nbank2 =n  
    i=0     
    while i!=nbank2:    
        flag=0
        for j in range(0,nbank2):
            if bank2[i][j]!=0:
                flag=1
        if flag ==0:
            for iterm in bank2:
                del iterm[i]
            del bank2[i]   
            i =i-1
            nbank2 = nbank2-1
        i=i+1   
    
        
    for i in range(0, m):
        for j in range(i+1 ,m):
            firm2[i][j]=0
            for k in range(0, n):
                firm2[i][j]=firm2[i][j]+((matrix[i][k])*(matrix[j][k]))/(tempFirm[i]*tempFirm[j])
                firm2[j][i]=firm2[i][j]   
#                if firm2[i][j]!=0:
#                    flag=1
#            if flag==0:
#                for iterm in firm2:
#                    del iterm[i]
#                del firm2[i]                
    
    mfirm2 =m  
    i=0     
    while i!=mfirm2:    
        flag=0
        for j in range(0,mfirm2):
            if firm2[i][j]!=0:
                flag=1
        if flag ==0:
            for iterm in firm2:
                del iterm[i]
            del firm2[i]   
            i =i-1
            mfirm2 = mfirm2-1
        i=i+1      
                
    return bank1,bank2,firm1,firm2,mfirm,nbank,indexBank,indexFirm        


def CalDist(m,n,alph):
    print u'\nCalDist###################################################\n'
    bank1,bank2,firm1,firm2,m,n,indexBank,indexFirm= MatrixTransform(m,n,matrix) 
    print '\nmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm',m,n
    d1Bank = []
    d2Bank = []
    dBank = []
    for i in range(0, n):
        d1Bank.append([])
        d2Bank.append([])
        dBank.append([])
        for j in range(0, n):
            d1Bank[i].append([])
            d2Bank[i].append([])
            dBank[i].append([])
            d1Bank[i][j] =  math.sqrt(2*(1-bank1[i][j]))
            d2Bank[i][j] = math.sqrt(2*(1-bank2[i][j]))
            dBank[i][j] = alph*d1Bank[i][j]+(1-alph)*d2Bank[i][j]
    d1Firm = []
    d2Firm = []
    dFirm = []
    for i in range(0, m):
        d1Firm.append([])
        d2Firm.append([])
        dFirm.append([])
        indexFirm.append([])
        for j in range(0, m):
            d1Firm[i].append([])
            d2Firm[i].append([])
            dFirm[i].append([])
            d1Firm[i][j] =  math.sqrt(2*(1-firm1[i][j]))
            d2Firm[i][j] = math.sqrt(2*(1-firm2[i][j]))
            dFirm[i][j] = alph*d1Firm[i][j]+(1-alph)*d2Firm[i][j] 
            
    print u'\n计算公司距离###################################################\n'
    filefirm = open('distfirm.txt','w')
    for i in range(0, m):
        if i!=0:
            print '\n',
            filefirm.write('\n')
        for j in range(0, m):  
            print dFirm[i][j], 
            filefirm.write(str(dFirm[i][j])+' ')
    filefirm.close()        
    print u'\n计算银行距离###################################################\n'  
    filebank = open('distbank.txt','w')
    for i in range(0, n):
        if (i !=0):      
            print '\n',
            filebank.write('\n')
        for j in range(0, n):  
            print dBank[i][j],
            filebank.write(str(dBank[i][j])+' ')
    filebank.close() 
    
    print u'\n01计算公司距离###################################################\n'
    filefirm01 = open('distfirm01.txt','w')
    for i in range(0, m):
        if i!=0:
            filefirm01.write('\n')
        for j in range(0, m):  
            filefirm01.write(str(firm1[i][j])+' ')
    filefirm01.close()  
    
    
    print u'\n公司01矩阵###################################################\n'
    filefirmMatrix01 = open('distfirmMatrix01.txt','w')
    for i in range(0, m):
        if i!=0:
            filefirmMatrix01.write('\n')
        for j in range(0, m):  
            if firm1[i][j]>0:
                filefirmMatrix01.write('1'+' ')
            else:
                filefirmMatrix01.write('0'+' ')
    filefirmMatrix01.close()  
    
    
    print u'\n01计算银行距离###################################################\n'  
    filebank01 = open('distbank01.txt','w')
    for i in range(0, n):
        if (i !=0):      
            filebank01.write('\n')
        for j in range(0, n):  
            filebank01.write(str(bank1[i][j])+' ')
    filebank01.close()  
    
    fileindexBank = open('indexBank.txt','w')    
    for i in range(0, n):
        fileindexBank.write(str(indexBank[i])+' ')
    fileindexBank.close()
    
    fileindexFirm = open('indexFirm.txt','w')    
    for i in range(0, m):
        fileindexFirm.write(str(indexFirm[i])+' ')
    fileindexFirm.close()    
    return dBank,dFirm    


if __name__ == '__main__':
    bank1=[]
    bank2=[]
    firm1=[]
    firm2=[]  
    matrix =[]
    indexBank =[]
    indexFirm =[]
    file_object = open('shouxin.txt') 
    allines = file_object.readlines()
    m=0
    n=0
    for line in allines:
        line = line.strip()
        List = line.split(' ')
     
        for i in range(0, len(List)): 
            t = float(List[i])
            List[i]=t    
        m=m+1    
        matrix.append(List)
        n=len(List)
        for i in range(0, m):
            #print '\n',
            for j in range(0, n):
                #print matrix[i][j],
                pass
    file_object.close()  
    
    for i in range(0, n):
        bank1.append([])
        bank2.append([])
        for j in range(0, n): 
            bank1[i].append([])
            bank2[i].append([]) 
    for i in range(0, m):
        firm1.append([])
        firm2.append([])
        for j in range(0, m):
            firm1[i].append([])
            firm2[i].append([])            
    bank1,bank2,firm1,firm2,m,n,indexBank,indexFirm = MatrixTransform(m,n,matrix)  
    print '\n','\nbank2#####################################################################',m,n
    for i in range(0, n):
        print '\n',
        for j in range(0, n): 
            print bank2[i][j],
    print '\n','#####################################################################',m,n
    for i in range(0, n):
        print '\n',
        for j in range(0, n): 
            print bank1[i][j],         
    print '\n','#####################################################################',m,n
    for i in range(0, m):
        print '\n',
        for j in range(0, m): 
            print firm1[i][j],
    print '\n','#####################################################################',m,n
    for i in range(0, m):
        print '\n',
        for j in range(0, m): 
            print firm2[i][j],          
    

    
#    dBank = []
#    dFirm = []
#    for i in range(0, nafter):
#        dBank.append([])
#        for j in range(0, nafter):
#            dBank[i].append([])     
#    for i in range(0, mafter):
#        dFirm.append([])
#        for j in range(0, mafter):
#            dFirm[i].append([])            
#    dBank,dFirm = CalDist(m,n,0.5)  

alph =0.5
print '\nmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm',m,n
d1Bank = []
d2Bank = []
dBank = []
for i in range(0, n):
    d1Bank.append([])
    d2Bank.append([])
    dBank.append([])
    for j in range(0, n):
        d1Bank[i].append([])
        d2Bank[i].append([])
        dBank[i].append([])
        d1Bank[i][j] =  math.sqrt(2*(1-bank1[i][j]))
        d2Bank[i][j] = math.sqrt(2*(1-bank2[i][j]))
        dBank[i][j] = alph*d1Bank[i][j]+(1-alph)*d2Bank[i][j]
d1Firm = []
d2Firm = []
dFirm = []
for i in range(0, m):
    d1Firm.append([])
    d2Firm.append([])
    dFirm.append([])
    indexFirm.append([])
    for j in range(0, m):
        d1Firm[i].append([])
        d2Firm[i].append([])
        dFirm[i].append([])
        d1Firm[i][j] =  math.sqrt(2*(1-firm1[i][j]))
        d2Firm[i][j] = math.sqrt(2*(1-firm2[i][j]))
        dFirm[i][j] = alph*d1Firm[i][j]+(1-alph)*d2Firm[i][j] 
        
print u'\n计算公司距离###################################################\n'
filefirm = open('distfirm.txt','w')
for i in range(0, m):
    if i!=0:
        print '\n',
        filefirm.write('\n')
    for j in range(0, m):  
        print dFirm[i][j], 
        filefirm.write(str(dFirm[i][j])+' ')
filefirm.close()        
print u'\n计算银行距离###################################################\n'  
filebank = open('distbank.txt','w')
for i in range(0, n):
    if (i !=0):      
        print '\n',
        filebank.write('\n')
    for j in range(0, n):  
        print dBank[i][j],
        filebank.write(str(dBank[i][j])+' ')
filebank.close() 

print u'\n01计算公司距离###################################################\n'
filefirm01 = open('distfirm01.txt','w')
for i in range(0, m):
    if i!=0:
        filefirm01.write('\n')
    for j in range(0, m):  
        filefirm01.write(str(firm1[i][j])+' ')
filefirm01.close()  

print u'\n公司01矩阵###################################################\n'
filefirmMatrix01 = open('distfirmMatrix01.txt','w')
for i in range(0, m):
    if i!=0:
        filefirmMatrix01.write('\n')
    for j in range(0, m):  
        if firm1[i][j]>0:
            filefirmMatrix01.write('1'+' ')
        else:
            filefirmMatrix01.write('0'+' ')
filefirmMatrix01.close()  

print u'\n01计算银行距离###################################################\n'  
filebank01 = open('distbank01.txt','w')
for i in range(0, n):
    if (i !=0):      
        filebank01.write('\n')
    for j in range(0, n):  
        filebank01.write(str(bank1[i][j])+' ')
filebank01.close()  


print u'\n银行01矩阵###################################################\n'
filebankMatrix01 = open('distbankMatrix01.txt','w')
for i in range(0, n):
    if i!=0:
        filebankMatrix01.write('\n')
    for j in range(0, n):  
        if bank1[i][j]>0:
            filebankMatrix01.write('1'+' ')
        else:
            filebankMatrix01.write('0'+' ')
filebankMatrix01.close()  

fileindexBank = open('indexBank.txt','w')    
for i in range(0, n):
    fileindexBank.write(str(indexBank[i])+' ')
fileindexBank.close()

fileindexFirm = open('indexFirm.txt','w')    
for i in range(0, m):
    fileindexFirm.write(str(indexFirm[i])+' ')
fileindexFirm.close()  
    

            
