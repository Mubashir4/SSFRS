# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:48:45 2021

@author: mubas
"""

import pandas as pd
import csv


file = 'rating.csv'
movie = 'movie_embeddings.csv'
rating = pd.read_csv(file,header=0)
rating['timestamp'] = pd.to_datetime(rating['timestamp'])
rating.sort_values(by=['userId','timestamp'],inplace=True)

movie_Embeddings = pd.read_csv(movie,header=None)
movie_Embeddings = movie_Embeddings.set_index(0).T.to_dict('list')

path = 'C:/Mubashir/Work/VLDB/Movie Lens/archive/Users/'

print('loading Done')

index = 0
for i in range(len(rating)):
    index += 1
    row = rating.iloc[i]
    emb = [row['movieId']]
    emb.extend(movie_Embeddings[row['movieId']])
    emb.extend([str(row['timestamp'])])
    emb.extend([row['rating']])
    with open(path+f'user_{row["userId"].csv}', mode='a+', newline='') as user_file:
        csv_writer = csv.writer(user_file,delimiter=',')
        csv_writer.writerow(emb)

            
            
        
    