# Keeping the random seed constant from one experiment to another makes it 
# easier to interpret the effects on hyper-parameters values 
from numpy.random import seed
seed(99)

import numpy as np

import keras
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Bidirectional
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.models import load_model
from keras import objectives

def create_model(input_dim, timesteps, intermediate_dim, batch_size, latent_dim, epochs, optimizer):
	# setup the network parameters:
	timesteps = timesteps
	input_dim = input_dim
	intermediate_dim = intermediate_dim
	batch_size = batch_size
	latent_dim = latent_dim
	epochs = epochs
	optimizer = optimizer
	
	# ----------------------- Encoder -----------------------
	inputs = Input(shape=(timesteps, input_dim,), name='encoder_input')

	# LSTM encoding
	h = Bidirectional(LSTM(intermediate_dim))(inputs) 

	# VAE Z layer
	mu = Dense(latent_dim)(h)
	sigma = Dense(latent_dim)(h)

	# reparametrization trick
	def sampling(args):
	  mu, sigma = args
	  batch     = K.shape(mu)[0]
	  dim       = K.int_shape(mu)[1]
	  eps       = K.random_normal(shape=(batch, dim))
	  return mu + K.exp(sigma / 2) * eps


	z = Lambda(sampling, output_shape=(latent_dim,))([mu, sigma])

	# instantiate the encoder model:
	encoder = Model(inputs, [mu, sigma, z], name='encoder')
	print(encoder.summary())
	# -------------------------------------------------------
	
	# ----------------------- Decoder -----------------------
	# decoded LSTM layer
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	h_decoded = RepeatVector(timesteps)(latent_inputs)
	h_decoded = Bidirectional(LSTM(intermediate_dim, return_sequences=True))(h_decoded) 
	# decoded layer
	outputs = Bidirectional(LSTM(input_dim, return_sequences=True))(h_decoded) 

	# instantiate the decoder model:
	decoder = Model(latent_inputs, outputs, name='decoder')
	print(decoder.summary())
	# -------------------------------------------------------
	
	# ----------------------- Classifier --------------------
	clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
	clf_intermediate = Dense(200, activation='relu')(clf_latent_inputs)
	clf_outputs = Dense(6, activation='softmax', name='class_output')(clf_intermediate)
	# instantiate the classifier model:
	clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')
	print(clf_supervised.summary())
	# -------------------------------------------------------
	
	# instantiate the VAE model:
	outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
	vae = Model(inputs, outputs, name='vae_mlp')
	print(vae.summary())

	# custom loss function:
	def vae_loss(x, x_decoded_mean):
		xent_loss = objectives.mse(x, x_decoded_mean)
		kl_loss = - 0.5 * K.mean(1 + sigma - K.square(mu) - K.exp(sigma))
		loss = xent_loss + kl_loss
		return loss	

	# compile the full model:
	vae.compile(optimizer=optimizer, loss={"decoder": vae_loss, "clf": "categorical_crossentropy"}, loss_weights={'decoder': 10, 'clf': 1.0})
	
	return vae
