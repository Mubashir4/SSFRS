# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:09:25 2021

@author: mubas
"""

import csv
from nltk.tokenize import word_tokenize


def read_csv(file):
    movie = {}
    gerne = {}
    gerne_ind_ = 0
    with open(file,encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i,row in enumerate(csv_reader):
            if(i!=0):
                ger = row[-1].split('|')
                for x in ger:
                    if(x not in gerne):
                        gerne_ind_ += 1
                        gerne[x] = gerne_ind_
                movie[row[0]] = [row[1],ger]
    return movie,gerne


def get_title_embedding(model,title,gerne):
    embeddings = {}
    for k,v in title.items():
        g = [0]*len(gerne)
        for j in v[1]:
            g[gerne[j]-1]=1
        embeddings[k] = model.infer_vector(word_tokenize(v[0])).tolist()
        embeddings[k].extend(g)
    return embeddings

#import movie titles and gernes here
file = 'movie.csv'
movie,gerne = read_csv(file)




from gensim.models.doc2vec import Doc2Vec, TaggedDocument
tagged_data = [TaggedDocument(s[0], [k]) for k,s in movie.items()]

model = Doc2Vec(tagged_data, vector_size = 10, window = 2, min_count = 1, epochs = 100, workers=4)
model.save("word2vec.model")


EMBEDDINGS = get_title_embedding(model,movie,gerne)

with open('movie_embeddings.csv', 'w', newline='') as csv_file:  
    writer = csv.writer(csv_file, delimiter=',')
    for key,value in EMBEDDINGS.items():
        writer.writerow([key].extend(value))
    































