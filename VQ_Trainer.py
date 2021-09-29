# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:41:59 2021

@author: mubas
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import VQ



def get_encoder(shape,latent_dim=15):
    encoder_inputs = keras.Input(shape = shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", name='enc_1')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name='enc_2')(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same", name='enc_output')(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(shape,latent_dim=15):
    latent_inputs = keras.Input(shape=get_encoder(shape).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name='dec_1')(latent_inputs)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name='dec_2')(x)
    print(x.shape)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same", name='dec_output')(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


"""
## Standalone VQ-VAE model
"""


def get_vqvae(shape, latent_dim=15, num_embeddings=64):
    vq_layer = VQ.VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(shape,latent_dim)
    decoder = get_decoder(shape,latent_dim)
    inputs = keras.Input(shape = shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")




class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance,shape, latent_dim=15, num_embeddings=64, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.shape = shape

        self.vqvae = get_vqvae( self.shape, self.latent_dim, self.num_embeddings)
        self.vqvae.summary()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
