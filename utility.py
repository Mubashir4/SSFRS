#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:07:45 2021

@author: s4523139
"""

import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from tensorflow.keras import utils
from sklearn import preprocessing

#########################################

def get_file_addresses(path):
    file_addresses = []
    for file_name  in os.listdir(path):
        if(file_name.endswith('.csv')):
            file_addresses.append([path,file_name.split('.')[0].split('_')[1]])
    print(len(file_addresses)," File Addresses Retrieved")
    return file_addresses
   
######################################### 
    
def import_LASTFM_files(file_):
    df = pd.read_csv(file_,header=None)
    df.rename(columns = {0:'id',1:'time',2:'f1',3:'f2',4:'f3',
                     4:'f3',5:'f4',6:'f5',7:'f6',8:'f7',
                     9:'f8',10:'f9',11:'f10'},inplace=True)
    df['time']= pd.to_datetime(df['time'])
    df['seconds'] = [(t-datetime.datetime(1970,1,1)).total_seconds() for t in df.time]
    norm = MinMaxScaler().fit(np.array(df['seconds']).reshape(-1,1))
    normalized_timestamp = norm.transform(np.array(df['seconds']).reshape(-1,1))
    normalized_timestamp = list(normalized_timestamp.reshape(-1))
    df['seconds'] = normalized_timestamp
    df.drop(['time'],axis=1,inplace=True)
    classes,df = Normalize_Item_Classes(df)
    print(f'user file loaded:  {file_}')
    return df,normalized_timestamp,classes

######################################### 
   
def split_data(df,sample_length,classes,to_cat = True):
  list_of_ids =  list(df.id)
  list_of_features = df[df.columns[0:]].values.tolist()
  if(to_cat):
      YF = labels_to_categorical(list_of_ids, classes)
  else:
      YF = list_of_ids
  
  X_id,X_fea,y = list(),list(),list()
  for i in range(len(list_of_ids)):
    last_ind = i+sample_length
    if(last_ind>len(list_of_ids)-1):
      break
    seq_id,seq_fea,seq_y = list_of_ids[i:last_ind],list_of_features[i:last_ind],YF[last_ind:last_ind+1]
    X_id.append(seq_id)
    X_fea.append(seq_fea)
    y.append(seq_y)
    
  X_id = np.array(X_id,dtype=int).reshape(len(X_id),len(X_id[0]))
  X_fea =  np.array(X_fea)
  if(to_cat):
      Labels = np.array(y)
      Labels = Labels[:, 0,:,]
  else:
      Labels =  np.squeeze(np.array(y,dtype=int).reshape(len(y),len(y[0])))  
  return X_id,X_fea,Labels

#########################################

def labels_to_categorical(labels,classes):
    return utils.to_categorical(labels, classes)


def generate_batches(_id_X,F,Y,batch_size):
  length = int((len(_id_X)/batch_size))
  batch_id_X,batch_F,batch_Y = list(),list(),list()
  for i in range(length):
    batch_id_X.append(_id_X[(i*batch_size):(i*batch_size)+batch_size])
    batch_F.append(F[(i*batch_size):(i*batch_size)+batch_size])
    batch_Y.append(Y[(i*batch_size):(i*batch_size)+batch_size])
  return batch_id_X,batch_F,batch_Y


#########################################

def get_Embedding_Seq(emb,lab,seq_size):
  seq = []
  new_lab = []
  for i in range(emb.shape[0]-1):
    if((i+seq_size)>(emb.shape[0]-1)):
      break
    else:
      seq.append(emb[i:i+seq_size])
      new_lab.append(lab[i+seq_size])
  return np.array(seq),np.array(new_lab)

#########################################


def Normalize_Item_Classes(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['id'])
    df['id'] = le.transform(df['id'])
    classes = df['id'].nunique()
    return classes,df

#########################################

# define a method to scale data, looping thru the columns, and passing a scaler
def scale_data(data, columns, scaler):
    for col in columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data

def import_MovieLens_files(Path,user):
    df = pd.read_csv(Path+f'user_{user}.csv',header=None)
    df.iloc[: , -2] = pd.to_datetime(df.iloc[: , -2])
    df.iloc[: , -2] = [(t-datetime.datetime(1970,1,1)).total_seconds() for t in df.iloc[: , -2]]
    IDs = df.iloc[:,0]
    df = df.iloc[: , 1:]
    Ratings = df.iloc[:,-1]
    df = df.iloc[: , :-1]
    #inorder to make 32 features
    df['32'] = df.iloc[:,-1]
    
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    
    return df,IDs,Ratings

######################################### 

def split_data_MovieLens(list_of_ids,df,sample_length,ratings):
  list_of_features = df[df.columns[0:]].values.tolist()
  YF = ratings
  X_id,X_fea,y = list(),list(),list()
  for i in range(len(list_of_ids)):
    last_ind = i+sample_length
    if(last_ind>len(list_of_ids)-1):
      break
    seq_id,seq_fea,seq_y = list_of_ids[i:last_ind],list_of_features[i:last_ind],YF[last_ind:last_ind+1]
    X_id.append(seq_id)
    X_fea.append(seq_fea)
    y.append(seq_y)
    
  
    
  X_id = np.array(X_id,dtype=int).reshape(len(X_id),len(X_id[0]))
  X_fea =  np.array(X_fea)
  y = np.array(y)
  return X_id,X_fea,y







