# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:06:57 2018

@author: sonu
"""

import keras


##make word to integer encoding
word_Encoding={}
dict_key=0 
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
        if word not in word_Encoding.keys():
            word_Encoding[word] = dict_key
            dict_key += 1            
        
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
        if word not in word_Encoding.keys():
            word_Encoding[word] = dict_key
            dict_key += 1
        
    else:
        if(len(sent)>0):
            x_valid_sent.append(sent)
            y_valid_sent.append(tag_Sent)
            sent=[]
            tag_Sent =[]
f.close()

x_test_sent =[]
y_test_sent =[]
y_test_sent2=[]
f= open('test.txt','r')
sent =[]
tag_Sent =[]
for line in f:
    line = line.rstrip()
    if(len(line)>0):
        word, tag = line.split()
        sent.append(word)
        tag_Sent.append(tag)
        if word not in word_Encoding.keys():
            word_Encoding[word] = dict_key
            dict_key += 1
        
    else:
        if(len(sent)>0):
            x_test_sent.append(sent)
            y_test_sent.append(tag_Sent)
            y_test_sent2.append(tag_Sent)
            sent=[]
            tag_Sent =[]
f.close()

word_Encoding['unk'] = 11311

##convert test sent to integers
y_train_sent += y_valid_sent
x_train_sent += x_valid_sent

x_train_copy = x_train_sent
x_test_copy =x_test_sent
y_train_copy = y_test_sent
y_test_copy = y_test_sent
dict_label={'O':0,'D':1,'T':2}

for i in range(len(y_train_sent)):
    for j in range(len(y_train_sent[i])):
        y_train_sent[i][j] = dict_label[y_train_sent[i][j]]
        x_train_sent[i][j] = word_Encoding[x_train_sent[i][j]]
for i in range(len(y_test_sent)):
    for j in range(len(y_test_sent[i])):
        y_test_sent[i][j] = dict_label[y_test_sent[i][j]]
        x_test_sent[i][j] = word_Encoding[x_test_sent[i][j]]



import numpy as np
from keras.preprocessing.sequence import pad_sequences
x_train_pad = pad_sequences(maxlen=110, sequences=x_train_sent, padding="post", value=11311)
x_test_pad = pad_sequences(maxlen=110, sequences=x_test_sent, padding="post", value=11311)
y_train_pad = pad_sequences(maxlen=110, sequences=y_train_sent, padding="post", value=0)
y_test_pad=pad_sequences(maxlen=110, sequences=y_test_sent, padding="post", value=0)



from keras.utils import to_categorical
y_train_cat=[to_categorical(i, num_classes=3) for i in y_train_pad]
y_test_cat = [to_categorical(i, num_classes=3) for i in y_test_pad]



##mdify
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional

from keras_contrib.layers import CRF

input = Input(shape=(110,))
model = Embedding(input_dim=len(word_Encoding), output_dim=100,
                  input_length=110, mask_zero=True)(input)  # 20-dim embedding

model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  
crf = CRF(len(dict_label))  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()
history = model.fit(x_train_pad, np.array(y_train_cat), batch_size=30, epochs=7,validation_split=0.05, verbose=1)

y_test_sent2=[]
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
            y_test_sent2.append(tag_Sent)
            sent=[]
            tag_Sent =[]
f.close()




from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
confusion_sent=[]
prec_sent =[]
f=open('resNER.txt','w')
for j in range(len(x_test_copy)):
    pred_output = model.predict(np.array([x_test_pad[j]]))
    dict_index = np.argmax(pred_output,axis=-1)
    dict_index=np.array(dict_index)
    #find actual length of x_test
    act_len = len(x_test_sent[j])
    pred_value=[]
    for i in range(act_len):
        
        f.write(list(word_Encoding.keys())[list(word_Encoding.values()).index(x_test_copy[j][i])])
        f.write(' ')
        pred_value.append(list(dict_label.keys())[list(dict_label.values()).index(dict_index[0][i])])
        f.write(list(dict_label.keys())[list(dict_label.values()).index(dict_index[0][i])])
        f.write('\n')
        
        conf_arg=[]
        for k in range(len(y_test_sent2[j])):
            conf_arg.append(y_test_sent2[j][k])
    confusion_sent.append(confusion_matrix(pred_value, conf_arg, labels=['O','D','T']))
    prec_sent.append(precision_recall_fscore_support(pred_value,conf_arg, average='micro'))


confusion_sum=np.sum(confusion_sent,axis=0)
print(confusion_sum)
recall_disease=confusion_sum[1,1]/np.sum(confusion_sum[:,1])
print('recall_disease',recall_disease)
precision_disease=confusion_sum[1,1]/np.sum(confusion_sum[1,:])
print('precision_disease',precision_disease)
recall_T=confusion_sum[2,2]/np.sum(confusion_sum[:,2])
print('recall_T',recall_T)
precision_T=confusion_sum[2,2]/np.sum(confusion_sum[2,:])
print('precision_T',precision_T)