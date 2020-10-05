import model
import utils
import keras 
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
	# Get data
	x_train, x_vali, x_test, y_train, y_vali, y_test, target_train, target_vali, target_test = utils.get_data()
	
	# Setup the network parameters:
	timesteps = x_train.shape[1]
	input_dim = x_train.shape[2]
	intermediate_dim = 300
	batch_size = 128
	latent_dim = 2
	epochs = 1000
	optimizer = keras.optimizers.Adam(learning_rate=0.0001)
	
	# Instantiate model
	RVAE = model.create_model(input_dim, 
			timesteps, 
			intermediate_dim, 
			batch_size, 
			latent_dim, 
			epochs, 
			optimizer)
	
	# Callbacks for training
	model_callbacks = utils.get_callbacks(RVAE, x_train, target_train)

	# Training:
	results = RVAE.fit(x_train, {'decoder': x_train, 'clf': y_train},
			shuffle=True,
			epochs=epochs,
			batch_size=batch_size,
			validation_data= (x_vali, {'decoder': x_vali, 'clf': y_vali}),
			callbacks=model_callbacks, verbose=2)
			
	# Results
	saved_VRAE_model = load_model('./models/VRAEBestModel.h5', compile=False)
	
	# Latent space
	encoder = saved_VRAE_model.layers[1]
	utils.viz_latent_space(encoder, x_test, target_test)
	
	# Classification	
	clf = saved_VRAE_model.layers[3]
	clfPred = clf.predict(encoder.predict(x_test))
	print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(clfPred, axis=1)))
	
	# Reconstruction
	decoder = saved_VRAE_model.layers[2]
	prediction = decoder.predict(encoder.predict(x_test), batch_size=batch_size)
	x_test_display = x_test.reshape(x_test.shape[0], x_test.shape[1])
	utils.displayReconstruction(x_test_display, prediction[:,:,0])