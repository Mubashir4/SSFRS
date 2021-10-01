#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:53:59 2021

@author: s4523139
"""

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


import csv

track_id = {}



with open('track_data.csv', 'r',encoding="utf8") as f:
    csv_reader = csv.reader(f, delimiter=',',quoting=csv.QUOTE_NONE)
    for i,row in enumerate(csv_reader):
        track_id[row[0]] = [row[1]+' '+row[2]]
  


sentences = {}  
len_m = 0    
largest_k = ''
for k,v in track_id.items():
    tokens = [word.lower() for word in v[0].split()]
    words = [word for word in tokens if word.isalpha()]
    sentences[k] = [w for w in words if not w in stop_words]
    if(len(sentences[k])>len_m):
        len_m=len(sentences[k])
        largest_k = k
    

tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(sentences.values())]
model = Doc2Vec(tagged_data, min_count=1,vector_size=10)
 
    
# train model
#model = Word2Vec(sentences.values(), min_count=1,vector_size=10)
#words = list(model.wv.key_to_index)
normed_vector = model.wv.get_vector("dish", norm=True) 
'''
for k,v in sentences.items():
    ar = [0.0]*(len_m*10)
    for i,w in enumerate(v):
        ar[(i*10):(i*10)+10] = model.wv.get_vector(w, norm=True) 
    with open('track_emb.csv', 'a',encoding="utf8",newline='') as g:
        csv_writer = csv.writer(g)
        csv_writer.writerow([k,ar])
    
'''    
dic_ = {}
for k,v in sentences.items():
    sen =v
    sen_emb = model.infer_vector(sen) 
    dic_[k] = sen_emb



file = 'Last_FM_Graphs/'
import os
path = file
graph_files = [f for f in os.listdir(path) if f.endswith('.csv')]

for f in graph_files:
    with open(file+f,'r') as h:
        csv_reader = csv.reader(h, delimiter=',')
        for row in csv_reader:
            with open(file+'Graph_with_features/'+f,'a',newline='') as graph:
                csv_writer = csv.writer(graph,delimiter=',')
                csv_writer.writerow([row[0],row[2]]+list(model.infer_vector(sentences[row[0]])))
                if(row[3]=='0'):
                    continue
                else:
                    csv_writer.writerow([row[1],row[3]]+list(model.infer_vector(sentences[row[1]])))

'''
for k,v in user_graphs.items():
    with open(file+str(k)+'.csv','w',newline='') as graph:
        csv_writer = csv.writer(graph,delimiter=',')
        for line in v:
            csv_writer.writerow(line)
        graph.close()
        print(str(k)+' written')   
'''



'''  
for k,v in sentences.items():
    sen =v
    for i,w in enumerate(v):
        sen_emb = model.infer_vector(sen) 
    with open('track_emb.csv', 'a',encoding="utf8",newline='') as g:
        csv_writer = csv.writer(g)
        csv_writer.writerow([k,sen_emb])    
  
'''    
  
    
  

  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    