import numpy as np
import math
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import backend as K
from sklearn.metrics import accuracy_score

N_CLASSES = 6

def convert_one_hot(x, target):
	'''
	Convert to one-hot encoding
	'''
	
	samples =  np.zeros((x.shape[0], N_CLASSES))
	for i in range(target.shape[0]):
		samples[i][int(target[i])] = 1
	return samples
	

def get_data():
	'''
	Get training/testing data 
	'''
	
	data = np.load('./data/arrhythmia_data.npy', allow_pickle=True).item()

	# Split into a train, vali, test
	x_train = data['input_train']
	x_vali = data['input_vali']
	x_test = data['input_test']
	 
	target_train = data['target_train']
	target_vali = data['target_vali']
	target_test = data['target_test']
	
	# Convert labels to one-hot encoding
	y_train = convert_one_hot(x_train, target_train)
	  
	y_vali = convert_one_hot(x_vali, target_vali)

	y_test = convert_one_hot(x_test, target_test)
	  
	return x_train, x_vali, x_test, y_train, y_vali, y_test, target_train, target_vali, target_test


def resultsTSC(model, RVAE=False):
	'''
	Accuracy report of the studied models 
	'''  

	# By class
	x_train, x_vali, x_test, y_train, y_vali, y_test, target_train, target_vali, target_test = get_data()

	# RVAE is passed as a tuple (encoder, clf)
	if type(model) == tuple:
		for i in range(len(np.unique(target_test))):
			data = x_test[np.where(target_test == i)]
			prediction = model[1].predict(model[0].predict(data))
			print("Accuracy for class", i ," :", accuracy_score(target_test[np.where(target_test == i)], np.argmax(prediction, axis=1)))
 
		prediction = model[1].predict(model[0].predict(x_test))

	else:
		for i in range(len(np.unique(target_test))):
			data = x_test[np.where(target_test == i)]
			prediction = model.predict(data)
			print("Accuracy for class", i ," :", accuracy_score(target_test[np.where(target_test == i)], np.argmax(prediction, axis=1)))
    
		prediction = model.predict(x_test)
    
	# Average 
	print("Accuracy: ", accuracy_score(target_test, np.argmax(prediction, axis=1)))


#------  CALLBACKS FOR TRAINING  ------#

class save_latent_space_viz(Callback):
	'''
	Saves a plot of the latent space on each epoch
	'''  
	
	def __init__(self, model, x_train, target_train):
		self.model = model
		self.x_test = x_train
		self.target_test = target_train
	
	def on_epoch_end(self, epoch, logs=None):
		encoder = self.model.layers[1]
		save_viz_latent_space(encoder, self.x_train, self.target_train, str(epoch))
	
def get_callbacks(model, x_train, target_train):
	'''
	Definition of the callbacks used
	''' 
	
	model_callbacks = [
		EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=60),
		ModelCheckpoint(filepath='VRAEBestModel.h5',monitor='val_loss', mode='min', verbose=1, save_best_only=True),
		TensorBoard(log_dir='./logs'),
		save_latent_space_viz(model, x_train, target_train)
	]
	
	return model_callbacks


def displayReconstruction(x_train, prediction):
	'''
	Displays a plot of the reconstruction of the data
	''' 
	
	data_size =  x_train.shape[0]
	selections = np.random.randint(data_size, size=5)
	x = np.arange(data_size)

	fig = plotly.subplots.make_subplots(rows = 2, cols = 5)
	colors = ['rgb(108, 225, 236)', 'rgb(108, 236, 150)', 'rgb(174, 108, 236)', 'rgb(218, 197, 26)', 'rgb(255, 178, 102)']

	for i in range(1, 6):
		trace = go.Scatter(
							x = x,
							y = x_train[selections[i-1]],
							name = 'Real'+str(i),
							marker=dict(
									color=colors[i-1]
							)
							)
		fig.append_trace(trace, 1, i)

	for i in range(1, 6):
		trace = go.Scatter(
							x = x,
							y = prediction[selections[i-1]],
							name = 'Reconstruction'+str(i),
							marker=dict(
									color=colors[i-1]
							)
							)
		fig.append_trace(trace, 2, i)

	iplot(fig)
 

def labels_to_string(labels):
	'''
	Pre-processing of the labels for plotly plot
	''' 
	
	final_target = []
	switcher = {
			0: "998na10",
			1: "998na30",
			2: "998na180",
			3: "999na10",
			4: "999na30",
			5: "999na180",
  }
	for i in range(len(labels)):
		final_target.append(switcher.get(labels[i]))
	return final_target


def viz_latent_space(encoder, data, target):
	'''
	Displays a plot of the latent space
	''' 
	
	mu = encoder.predict(data)
	plt.figure(figsize=(8, 10))
	plt.scatter(mu[:, 0], mu[:, 1],c=target)
	plt.xlabel('z - dim 1')
	plt.ylabel('z - dim 2')
	plt.colorbar()
	plt.show()


def save_viz_latent_space(encoder, data, target, epoch):
	'''
	Saves a plot of the latent space
	''' 
	
	mu = encoder.predict(data)
	plt.figure(figsize=(8, 10))
	plt.scatter(mu[:, 0], mu[:, 1],c=target)
	plt.xlabel('z - dim 1')
	plt.ylabel('z - dim 2')
	plt.colorbar()
	plt.savefig('./images/latent_space'+epoch+'.png')


def viz_latent_space_plotly(encoder, data, target, epoch):
	'''
	Shows a plot of the latent space using plotly
	''' 
	
	import plotly.express as px
	import pandas as pd
  
	target = labels_to_string(target)
	mu = encoder.predict(data)
	data = {'z - dim 1': mu[:, 0], 'z - dim 2':mu[:, 1], 'label': target}
	df = pd.DataFrame(data, columns = ['z - dim 1', 'z - dim 2', 'label'])
	fig = px.scatter(df, x="z - dim 1", y="z - dim 2", color="label", width=600, height=800)
	fig.update_layout({
	'plot_bgcolor': 'rgba(0, 0, 0, 0)',
	'paper_bgcolor': 'rgba(0, 0, 0, 0)'
	})
	fig.show()


# Find the optimum learning rate (source: https://www.kaggle.com/paultimothymooney/learning-rate-finder-for-keras)
class MetricsCheckpoint(Callback):
	'''
	Callback that saves metrics after each epoch
	''' 

	def __init__(self, savepath):
		super(MetricsCheckpoint, self).__init__()
		self.savepath = savepath
		self.history = {}
	def on_epoch_end(self, epoch, logs=None):
		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)
		np.save(self.savepath, self.history)


def plotKerasLearningCurve():
	'''
	Visualize Cyclical learning curve
	''' 
	
	plt.figure(figsize=(10,5))
	metrics = np.load('logs.npy')[()]
	filt = ['acc'] # try to add 'loss' to see the loss learning curve
	for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
			l = np.array(metrics[k])
			plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
			x = np.argmin(l) if 'loss' in k else np.argmax(l)
			y = l[x]
			plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
			plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
	plt.legend(loc=4)
	plt.axis([0, None, None, None]);
	plt.grid()
	plt.xlabel('Number of epochs')


class LRFinder:
	'''
	Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
	See for details:
	https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
	'''
    
	def __init__(self, model):
		self.model = model
		self.losses = []
		self.lrs = []
		self.best_loss = 1e9

	def on_batch_end(self, batch, logs):
		# Log the learning rate
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)

		# Log the loss
		loss = logs['loss']
		self.losses.append(loss)

		# Check whether the loss got too large or NaN
		if math.isnan(loss) or loss > self.best_loss * 4:
			self.model.stop_training = True
			return

		if loss < self.best_loss:
			self.best_loss = loss

			# Increase the learning rate for the next batch
			lr *= self.lr_mult
			K.set_value(self.model.optimizer.lr, lr)

	def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
		num_batches = epochs * x_train.shape[0] / batch_size
		self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)

		# Save weights into a file
		self.model.save_weights('tmp.h5')

		# Remember the original learning rate
		original_lr = K.get_value(self.model.optimizer.lr)

		# Set the initial learning rate
		K.set_value(self.model.optimizer.lr, start_lr)

		callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

		self.model.fit(x_train, y_train,
									batch_size=batch_size, epochs=epochs,
									callbacks=[callback])

		# Restore the weights to the state before model fitting
		self.model.load_weights('tmp.h5')

		# Restore the original learning rate
		K.set_value(self.model.optimizer.lr, original_lr)

	def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
		'''
		Plots the loss.
		Parameters:
			n_skip_beginning - number of batches to skip on the left.
			n_skip_end - number of batches to skip on the right.
		'''
        
		plt.ylabel("loss")
		plt.xlabel("learning rate (log scale)")
		plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
		plt.xscale('log')

	def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
		'''
		Plots rate of change of the loss function.
		Parameters:
		sma - number of batches for simple moving average to smooth out the curve.
		n_skip_beginning - number of batches to skip on the left.
		n_skip_end - number of batches to skip on the right.
		y_lim - limits for the y axis.
		'''
        
		assert sma >= 1
		derivatives = [0] * sma
		for i in range(sma, len(self.lrs)):
			derivative = (self.losses[i] - self.losses[i - sma]) / sma
			derivatives.append(derivative)

		plt.ylabel("rate of loss change")
		plt.xlabel("learning rate (log scale)")
		plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
		plt.xscale('log')
		plt.ylim(y_lim)