# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 13:52:35 2021

@author: mubas
"""

import numpy as np
import utility as utl
import csv
import glob

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import VQ_Trainer
from tensorflow.keras.callbacks import EarlyStopping
import time



class User:
    def __init__(self,path,file,seq_size,batch_size,client_server_directory = 'client_server_directory/'):
        
        self.path = path
        self.file = file
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.client_server_directory = client_server_directory
        
        df,IDs,Ratings = utl.import_MovieLens_files(self.path,self.file)
        IDs,df,Ratings  = utl.split_data_MovieLens(IDs,df,seq_size,Ratings)
        self.data_variance = np.var(df)
        self.batch_id_X,self.batch_F,self.batch_Y = utl.generate_batches(IDs,df,Ratings,batch_size)
        
        self.current_iteration = -1
        self.time_ = 0
        
    def get_recommendations(self):
        self.current_iteration += 1
        if(self.current_iteration<=(len(self.batch_id_X)-1)):
            if(self.current_iteration==0):
                F_train = np.expand_dims(self.batch_F[self.current_iteration], -1)
                seq_shape = (F_train[0].shape)
                self.vqvae_trainer = VQ_Trainer.VQVAETrainer(self.data_variance, seq_shape,latent_dim=15, num_embeddings=64)
                self.vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
                start = time.time()
                self.vqvae_trainer.fit(F_train, epochs=100, callbacks=[self.callback()])
                end = time.time()
            else:
                F_train = np.expand_dims(self.batch_F[self.current_iteration], -1)
                start = time.time()
                self.vqvae_trainer.fit(F_train, epochs=100, callbacks=[self.callback()])  
                end = time.time()
            self.time_ += (end-start)
        else:
            print(f'All Training batches executed for user {self.file}')
    
    def get_weights(self,layer='encoder'):
        self.w = self.vqvae_trainer.vqvae.get_layer(layer).get_weights()
        return  self.w
    
    def set_weights(self,w,layer='encoder'):
        self.vqvae_trainer.vqvae.get_layer(layer).set_weights(w)
        self.w = w
        print('weights updated')
      
    def callback(self,monitoring='vqvae_loss',  patience = 10,  restore_best_weights = True, verbose = 1,mode='auto'):
        return EarlyStopping(monitor=monitoring, mode='min',min_delta=0.0001,baseline=None, patience = patience, restore_best_weights = restore_best_weights, verbose=verbose)
    
    def write_weights(self):
        file_name = self.client_server_directory+f'client_{self.file}_{self.current_iteration}.csv'
        if(os.path.isfile(file_name)):
            os.remove(file_name)
            with open(file_name,mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(self.get_weights())
        else:
            pre_iteration = self.client_server_directory+f'client_{self.file}_*.csv'
            x = glob.glob(pre_iteration)
            if(file_name in x):
                m=x[-1].split('_')[-1].split('.')[0]
                self.current_iteration = int(m)
                with open(x[-1],mode='r') as f:
                    reader = csv.reader(f)
                    for w in reader:
                        self.set_weights(w)
                        break
                print('Previous Iteration not processessed... reverting to the weights of the saved file')
            else:
                with open(file_name,mode='w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.get_weights())
                    
    def read_weights(self):
        file_name = self.client_server_directory+f'server_{self.file}_{self.current_iteration}.csv'
        if(os.path.isfile(file_name)):
            with open(file_name,mode='r') as f:
                reader = csv.reader(f)
                for w in reader:
                    self.set_weights(w)
                    break
        else:
            print('weights not avalible yet')
                    
                
    def write_Embeddings(self):
        encoder = self.vqvae_trainer.vqvae.get_layer("encoder")   
        F = self.batch_F
        Y = self.batch_Y
        for i,f in enumerate(F):
            if i==0:
                x=f
                y=Y[i]
            else:
                x = np.concatenate((x,f),axis=0)
                y = np.concatenate((y,Y[i]),axis=0)
        F_train = np.expand_dims(x, -1)
        embeddings = encoder.predict(F_train)
        np.save(f'Embeddings_User_{self.file}',embeddings)
        np.save(f'Ratings_User_{self.file}',y)
        
            
            
        