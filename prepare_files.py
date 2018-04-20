# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:24:29 2018

@author: sonu
"""

f2=open('valid.txt','a')
f3=open('train.txt','a')
f4 = open('test.txt','a')
lineNum=0
with open('tsvNER.txt') as f:
        for line in f:
            line = line.rstrip()
            lineNum +=1
            if lineNum<9099:
                if(len(line)>0):
                    word, tag = line.split()
                    f2.write(word+'\t'+tag+'\n')
                else:
                    f2.write('\n')
#                f2.write(line)
                
            elif lineNum<20963:
                if(len(line)>0):
                    word, tag = line.split()
                    f4.write(word+'\t'+tag+'\n')
                else:
                    f4.write('\n')
            else:
                if(len(line)>0):
                    word, tag = line.split()
                    f3.write(word+'\t'+tag+'\n')
                else:
                    f3.write('\n')
                
f2.close()
f3.close()
f4.close()