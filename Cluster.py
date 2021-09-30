# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:33:59 2021

@author: mubas
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Clustring:
    
    def __init__(self,_input):
        self.input = _input
        self.modified_input = self.shape_input()
        self.latent_dim = 10
        self.Encoder()
        self.Decoder()
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=keras.optimizers.Adam())
        self.vae.fit(self.modified_input, epochs=30, batch_size=128)
        self.z_mean, _, _ = self.vae.encoder.predict(self.modified_input)
        self.x = self.compute_optimal_number_of_clusters(20)
        self.compute_Cluster_first(self.x)
        
        
        
    def shape_input(self):
        modified_input = []
        for k,v in self.input.items():
            w1 = v[0]
            w2 = v[2]
            w3 = v[4]
            w1,w2,w3 = w1.flatten(),np.append(w2.flatten(),w2.flatten()[0:32]),w3.flatten()
            w = np.concatenate((w1, w2, w3), axis=0)
            w = w.reshape(616,-1, 1)
            modified_input.append(w)
        return np.array(modified_input)
    
    def Encoder(self):
        encoder_inputs = keras.Input(shape=(616,32, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()
        
    def Decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(78848, activation="relu")(latent_inputs)
        x = layers.Reshape((154, 8, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        self.decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = keras.Model(latent_inputs, self.decoder_outputs, name="decoder")
        self.decoder.summary()
        
        
        
    def update_weights(self,labels,user_dic,avg):
        for i,m in enumerate(user_dic.items()):
            k,m = m
            l = labels[i]
            m.set_weights(avg[l])
        return user_dic
    
        
                
    
    def average(self,avg,avg_count):
        for i in range(len(avg)):
            if(len(avg[i])!=0):
                avg[i][0] = avg[i][0]/avg_count[i]
                avg[i][1] = avg[i][1]/avg_count[i]
                avg[i][2] = avg[i][2]/avg_count[i]
                avg[i][3] = avg[i][3]/avg_count[i]
                avg[i][4] = avg[i][4]/avg_count[i]
                avg[i][5] = avg[i][5]/avg_count[i]
        return avg
            
            
    def weight_aggregator(self,labels,user_weights,num_clusters):
        self.avg = [[]]*num_clusters
        avg_count = [0]*num_clusters
        
        for i,w in enumerate(user_weights.items()):
            k,w = w
            l = labels[i]
            if self.avg[l] == []:
                self.avg[l] = w
            else:
                self.avg[l][0]+=w[0]
                self.avg[l][1]+=w[1]
                self.avg[l][2]+=w[2]
                self.avg[l][3]+=w[3]
                self.avg[l][4]+=w[4]
                self.avg[l][5]+=w[5]
            avg_count[labels[i]]+=1
        return self.average(self.avg,avg_count),avg_count
        
    def compute_Cluster_first(self,k):
        self.kmean = KMeans(n_clusters=k, random_state=0)
        km = self.kmean.fit(self.z_mean)
        self.labels = km.labels_
        
    
    def reshape_input(self,v):
        w1 = v[0]
        w2 = v[2]
        w3 = v[4]
        w1,w2,w3 = w1.flatten(),np.append(w2.flatten(),w2.flatten()[0:32]),w3.flatten()
        w = np.concatenate((w1, w2, w3), axis=0)
        w = w.reshape(616,-1, 1)
        return np.array([w])
    
    def compute_Cluster(self,model):
        emb, _, _  = self.vae.encoder.predict(self.reshape_input(model))
        label = self.kmean.predict(emb)
        return label
        
    def compute_optimal_number_of_clusters(self,range_):
        model = KElbowVisualizer(KMeans(), k=range_, timings=False,show=False)
        model.ax.set(title='')
        model.fit(self.z_mean)
        #model.fig.show()
        #model.ax.legend().set_visible(False)
        return model.elbow_value_
        
 
    def recompute_clusters(self,input_):
        self.input = input_
        self.modified_input = self.shape_input()
        self.z_mean, _, _ = self.vae.encoder.predict(self.modified_input)
        x = self.compute_optimal_number_of_clusters(20)
        self.compute_Cluster_first(x)
        
        
        
      #metric='silhouette',  
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    