# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:12:19 2018

@author: sonu
"""
import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=False)

x_train =[]
y_train =[]
f= open('train.txt','r')
for line in f:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        x_train.append(word)
        y_train.append(tag)
f.close()

f= open('valid.txt','r')
x_valid=[]
y_valid=[]
for line in f:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        x_valid.append(word)
        y_valid.append(tag)

f.close()	
# ap,pa gives error, pa is giving best res
algorithm = {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
graphModel={'crf1d',}
trainer.select('pa',type='crf1d')
trainer.append(x_train,y_train)
trainer.append(x_valid,y_valid,group=1)
#parameter setting



trainer.train( 'pycrfmodel')

tagger = pycrfsuite.Tagger()
tagger.open('pycrfmodel')

x_test =[]
y_test =[]
f2= open('test.txt','r')
for line in f2:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        x_test.append(word)
        y_test.append(tag)

#myTest=x_test[0:2]
predict = tagger.tag(x_test)
def accuracy(predict,actual):
    correct=0
    for i in range(len(predict)):
        if(predict[i]==actual[i]):
            correct +=1
    print('correct =',correct)
    print('accuracy is:',correct*100/len(predict))


accuracy(predict,actual=y_test)    

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predict,labels=['O','D','T'])

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, predict, average='macro')


#### training using sentence models
x_train_sent =[]
y_train_sent =[]
f= open('train.txt','r')
sent =[]
tag_Sent =[]
for line in f:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        sent.append(word)
        tag_Sent.append(tag)
        
    else:
        if(len(sent)>0):
            x_train_sent.append(sent)
            y_train_sent.append(tag_Sent)
            sent=[]
            tag_Sent =[]
f.close()

x_valid_sent =[]
y_valid_sent =[]
f= open('valid.txt','r')
sent =[]
tag_Sent =[]
for line in f:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        sent.append(word)
        tag_Sent.append(tag)
        
    else:
        if(len(sent)>0):
            x_valid_sent.append(sent)
            y_valid_sent.append(tag_Sent)
            sent=[]
            tag_Sent =[]
f.close()

# ap,pa gives error, pa is giving best res
algorithm = {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
graphModel={'crf1d',}
trainer.select('pa',type='crf1d')

for i in range(len(x_train_sent)):
    trainer.append(x_train_sent[i],y_train_sent[i])
for i in range(len(x_valid_sent)):
    trainer.append(x_valid_sent[i],y_valid_sent[i],group=1)
#parameter setting



trainer.train( 'pycrfmodel')

tagger = pycrfsuite.Tagger()
tagger.open('pycrfmodel')

x_test_sent =[]
y_test_sent =[]
f= open('test.txt','r')
sent =[]
tag_Sent =[]
for line in f:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        sent.append(word)
        tag_Sent.append(tag)
        
    else:
        if(len(sent)>0):
            x_test_sent.append(sent)
            y_test_sent.append(tag_Sent)
            sent=[]
            tag_Sent =[]
f.close()

#myTest=x_test[0:2]
predict=[]
for i in range(len(x_test_sent)):
    predict.append(tagger.tag(x_test_sent[i]))

def accuracy(predict,actual):
    correct=0
    t=0
    for i in range(len(predict)):
        t+=len(predict[i])
        for j in range(len(predict[i])):
            if(predict[i][j]==actual[i][j]):
                correct +=1
    print('correct =',correct)
    print('accuracy is:',correct*100/t)


accuracy(predict,actual=y_test_sent)    

act =[]
pred =[]
for i in range(len(predict)):
    for j in range(len(predict[i])):
        act.append(y_test_sent[i][j])
        pred.append(predict[i][j])

from sklearn.metrics import confusion_matrix
confusion_matrix(act,pred,labels=['O','D','T'])

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(act, pred, average='macro')




