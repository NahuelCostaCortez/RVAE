'''
Different progressions of atrial fibrillation can be simulated with this class
'''

import sys
import numpy as np
from numpy.random import seed
from math import ceil

MAX_SIMULATION_TIME = 365*10
SIMULATION_SIZE = 14000
SERIES_SIZE = 146
N_CLASSES = 6
RANDOM_SEED = 99


def split(samples, proportions):
	"""
	Return train/validation/test split.
	"""
	np.random.seed(RANDOM_SEED)

	n_total = samples.shape[0]
	n_train = ceil(n_total*proportions[0])
	n_test = ceil(n_total*proportions[2])
	n_vali = n_total - (n_train + n_test)
	
	# permutation to shuffle the samples
	shuff = np.random.permutation(n_total)
	train_indices = shuff[:n_train]
	vali_indices = shuff[n_train:(n_train + n_vali)]
	test_indices = shuff[(n_train + n_vali):]
	
	# split up the samples
	train = samples[train_indices]
	vali = samples[vali_indices]
	test = samples[test_indices]

	return train, vali, test

def AF_dataset():
	"""
	Build dataset from each class
	"""

	data_classes = [(10, 0.998), (30, 0.998), (180, 0.998), (10, 0.999), (30, 0.999), (180, 0.999)]
	data = np.empty([SIMULATION_SIZE, SERIES_SIZE, 1])

	for c in data_classes:
		data_path = './data/markov_'+str(c[1])[2:]+'na'+str(c[0])+'.npy'
		class_data = np.load(data_path, allow_pickle=True)
		data = np.row_stack((data, class_data))

	args = (0*np.ones(int(len(data)/N_CLASSES)), 1*np.ones(int(len(data)/N_CLASSES)), 2*np.ones(int(len(data)/N_CLASSES)), 3*np.ones(int(len(data)/N_CLASSES)), 4*np.ones(int(len(data)/N_CLASSES)), 5*np.ones(int(len(data)/N_CLASSES)))
	target= np.concatenate(args)
	
	input_train, input_vali, input_test = split(data, [0.6, 0.2, 0.2])
	target_train, target_vali, target_test = split(target, [0.6, 0.2, 0.2])

	dict_data = {'input_train': input_train, 'input_vali': input_vali, 'input_test': input_test, 'target_train': target_train, 'target_vali': target_vali, 'target_test': target_test}
	np.save('./data/arrhythmia_data.npy', dict_data)


def markov_samples(num_examples, betaNA, alpha):
	'''
	Simulation of a patient's ECG
	'''

	# all times in days
	lambdaGA = 1.0/(3.0/24) # 3 hours
	lambdaA0 = 1.0/2.0 # 2 days
	pAN = 0.75
	pAG = 1 - pAN
	lambdaNA = 1.0/betaNA # 6 months

	ENORMAL = 0
	EGAP = 1
	EARRHYTHMIA = 2
	samples = []

	for i in range(num_examples):
		# Markov model simulation
		state = ENORMAL
		day = 0.0
		events = []
		times = [0]
		stateList = [state]
		stateListTime = {day: state}
		while True:
			if state == ENORMAL:
				betaNA = 1.0/lambdaNA
				delay = np.random.exponential(betaNA)
				nextState = EARRHYTHMIA
			elif state == EARRHYTHMIA:
				lA = lambdaA0 * (alpha**day)
				mu2 = lA * pAN
				l1 = lA * pAG
				betaAN = 1.0/mu2
				betaAG = 1.0/l1
				delayAN = np.random.exponential(betaAN)
				delayAG = np.random.exponential(betaAG)
				if delayAN<delayAG:
					nextState = ENORMAL
					delay = delayAN
				else:
					nextState = EGAP
					delay = delayAG
			else:
				betaGA = 1.0/lambdaGA
				delay = np.random.exponential(betaGA)
				nextState = EARRHYTHMIA
			day = day + delay
			state = nextState

			if nextState==EARRHYTHMIA:
				events = events + [day]
				
			times = times + [day]
			stateList = stateList + [nextState]
			stateListTime[day]=nextState

			if day>MAX_SIMULATION_TIME:
				break

		days=[]
		for day in range(MAX_SIMULATION_TIME): 
			days.append(day)

		porcentaje = np.zeros(len(days))

		# percentage of time in arrhythmia per day
		keys=list(stateListTime.keys())
		for i in days:
			for day in range(len(keys)):
				# if it is the last day that an event is scheduled, there is no need to compare to the next day,
 				# then the current day is marked with the last change of state on record
				if day == len(keys):
					if i > int(keys[day]):
						if stateListTime[keys[day]] == 2:
							porcentaje[i]=1

				# if it's not the last day in which there was a change of state, check if in the current day 
				# there was any change of state
				elif i == int(keys[day]):
					# if state 2
					if stateListTime[keys[day]] == 2:
						# calculate the percentage of time at state 2
						hours = abs(keys[day]) - abs(int(keys[day]))
						# if the next change of state happens on the same day, calculate the percentage of time on arryhtmia that day
						if int(keys[day+1]) == int(keys[day]):
							hours2 = abs(keys[day+1]) - abs(int(keys[day+1]))
							if porcentaje[i]>0:
								porcentaje[i]+=((hours2-hours)/1)
							else:
								porcentaje[i]=(hours2-hours)/1
						# if the next change of status happens on another day, it means that the 100% of time that day 
						# the patient was in arrhthmia, unless he/she were already on arrhythmia that day
						else:
							if porcentaje[i]>0:
								hours = abs(keys[day]) - abs(int(keys[day]))
								porcentaje[i]+=1-hours
							else:
								porcentaje[i]=1-(abs(keys[day]) - abs(int(keys[day])))
				# if there is no change in status on this day it may or may not be in arrhythmia, 
				# it is checked by looking at the last change of state
				else:
					if i > int(keys[day]) and i< int(keys[day+1]):
						# if the last state marked is arrhythmia, the patient is in arrhythmia state
						if stateListTime[keys[day]] == 2:
							porcentaje[i]=1
						# if the last state marked was not arrhythmia, the patient is in normal state, the percentage is 0

		# the values are smoothed
		n_points = len(days)
		x_vals = np.arange(n_points)
		y_vals=porcentaje
		def fwhm2sigma(fwhm):
			return fwhm / np.sqrt(8 * np.log(2))
		FWHM = MAX_SIMULATION_TIME/SERIES_SIZE * 10
		sigma = fwhm2sigma(FWHM)
		smoothed_vals = np.zeros(y_vals.shape)
		for x_position in x_vals:
			kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
			kernel = kernel / sum(kernel)
			smoothed_vals[x_position] = sum(y_vals * kernel)

		# finally, the time series is scaled to another with much fewer points
		scale = MAX_SIMULATION_TIME/SERIES_SIZE
		values = np.zeros(int(len(smoothed_vals)/scale))
		cont = 0
		pos = 0
		i = 0
		while i < len(smoothed_vals):
			while cont<scale:
				values[pos] += smoothed_vals[i]
				i += 1
				cont += 1
			if cont==scale:
				cont = 0
				values[pos] = values[pos]/scale
				pos += 1
		samples.append(values)	

	samples = np.array(samples)
	samples = samples.reshape(samples.shape[0],samples.shape[1],1)
	return samples

# Labels: F1A0.5,F2A0.5,F3A0.5,F4A0.5,F3A0.125,F3A0.25,F3A0.375,F3A0.5
def sine_wave(seq_length=144, num_samples=28*5*100, num_signals=1):
  frequency = np.array([1,2,3,4,3,3])
  amplitude = np.array([0.5,0.5,0.5,0.375,0.25,0.125])
  num_classes = len(frequency)
  ix = np.arange(seq_length) + 1
  samples = []
  labels = []
  for i in range(num_classes):
    for j in range(int(num_samples/num_classes)):
      signals = []
      f = frequency[i]     # frequency
      A = amplitude[i]       # amplitude
      # offset
      offset = np.random.uniform(low=-np.pi, high=np.pi)
      signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset) + 0.5)
      samples.append(np.array(signals).T)
  # the shape of the samples is num_samples x seq_length x num_signals
  samples = np.array(samples)

  # target data
  n_classes = 6
  args = (0*np.ones(int(len(data)/n_classes)), 1*np.ones(int(len(data)/n_classes)), 2*np.ones(int(len(data)/n_classes)), 3*np.ones(int(len(data)/n_classes)), 4*np.ones(int(len(data)/n_classes)), 5*np.ones(int(len(data)/n_classes)))
  target = np.concatenate(args)
  
  # shuffle the data
  randomize = np.arange(len(target))
  np.random.shuffle(randomize)
  target = target[randomize]
  data = data[randomize]

  # Split in train/test input/target
  input_train, input_vali, input_test = split(data, [0.6, 0.2, 0.2])
  target_train, target_vali, target_test = split(target, [0.6, 0.2, 0.2])

  # Save the generated data
  dict_data = {'input_train': input_train, 'input_vali': input_vali, 'input_test': input_test, 'target_train': target_train, 'target_vali': target_vali, 'target_test': target_test}
  np.save('./sine_wave.npy', dict_data)

# this may take a while, consider parallelization
data_classes = [(10, 0.998), (30, 0.998), (180, 0.998), (10, 0.999), (30, 0.999), (180, 0.999)]
for c in data_classes:
	samples = markov_samples(SIMULATION_SIZE, c[0], c[1])
	data_path = './data/markov_'+str(c[1])[2:]+'na'+str(c[0])+'.npy'
	np.save(data_path, samples)
	print('Saved training data to', data_path)

AF_dataset()
sine_wave()
