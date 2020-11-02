# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2019
# Assignment 5
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import tensorflow as tf
import numpy as np
import random
from utils import *



# Function to get word2vec representations
#
# Arguments:
# reviews: A list of strings, each string represents a review
#
# Returns: mat (numpy.ndarray) of size (len(reviews), dim)
# mat is a two-dimensional numpy array containing vector representation for ith review (in input list reviews) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
def w2v_rep(reviews):
	dim = 300
	mat = np.zeros((len(reviews), dim))
	# [YOUR CODE HERE]
	# load embeddings
	word2vec = load_w2v()
	# iterate through list of reviews
	for review in range(0, len(reviews)):
		# for each review, convert it into tokens
		tokens = get_tokens(reviews[review])
		embedding = np.zeros(dim)
		# for each index of the embedding
		for i in range(0, dim):
			num_valid = 0
			# for each token
			for j in range(0, len(tokens)):
				# if the token is in the vocabulary
				if tokens[j] in word2vec.keys():
					# access the i-th index at the corresponding vector
					# add it to the running sum
					embedding[i] += word2vec[tokens[j]][i]
					num_valid += 1
			mat[review][i] = embedding[i] / num_valid
	return mat


# Function to build a feed-forward neural network using tf.keras.Sequential model. You should build the sequential model
# by stacking up dense layers such that each hidden layer has 'relu' activation. Add an output dense layer in the end
# containing 1 unit, with 'sigmoid' activation, this is to ensure that we get label probability as output
#
# Arguments:
# params (dict): A dictionary containing the following parameter data:
#					layers (int): Number of dense layers in the neural network
#					units (int): Number of units in each dense layer
#					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#
# Returns:
# model (tf.keras.Sequential), a compiled model created using the specified parameters
def build_nn(params):
	model = tf.keras.Sequential()
	# [YOUR CODE HERE]
	for _ in range(0, params['layers']):
		# add a hidden layer
		model.add(tf.keras.layers.Dense(params['units'], activation='relu'))
	# add the output layer
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=['accuracy'])
	return model


# Function to select the best parameter combination based on accuracy by evaluating all parameter combinations
# This function should train on the training set (X_train, y_train) and evluate using the validation set (X_val, y_val)
#
# Arguments:
# params (dict): A dictionary containing parameter combinations to try:
#					layers (list of int): Each element specifies number of dense layers in the neural network
#					units (list of int): Each element specifies the number of units in each dense layer
#					loss (list of string): Each element specifies the type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (list of string): Each element specifies the type of optimizer to use while training ('sgd' or 'adam')
#					epochs (list of int): Each element specifies the number of iterations over the training set
# X_train (numpy.ndarray): A matrix containing w2v representations for training set of shape (len(reviews), dim)
# y_train (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_train of shape (X_train.shape[0], )
# X_val (numpy.ndarray): A matrix containing w2v representations for validation set of shape (len(reviews), dim)
# y_val (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_val of shape (X_val.shape[0], )
#
# Returns:
# best_params (dict): A dictionary containing the best parameter combination:
#	    				layers (int): Number of dense layers in the neural network
#	 	     			units (int): Number of units in each dense layer
#	 					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#						optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#						epochs (int): Number of iterations over the training set
def find_best_params(params, X_train, y_train, X_val, y_val):
	best_params = dict()

	# Note that you don't necessarily have to use this loop structure for your experiments
	# However, you must call reset_seeds() right before you call build_nn for every parameter combination
	# Also, make sure to call reset_seeds right before every model.fit call

	# Get all parameter combinations (a list of dicts)
	# [YOUR CODE HERE]

	param_combinations = []

	keys = params.keys()
	values = (params[key] for key in keys)
	param_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

	# replace 0 at 'max' with highest accuracy, {} at 'set' with the combination
	best_params = {'max': 0, 'params': {}}
	# Iterate over all combinations using one or more loops
	for param_combination in param_combinations:
    	# Reset seeds and build your model
		reset_seeds()
		model = build_nn(param_combination)
		# Train and evaluate your model, make sure you call reset_seeds before every model.fit call
		# [YOUR CODE HERE]
		reset_seeds()
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		model.fit(X_train, y_train, epochs=param_combination['epochs'])

		reset_seeds()
		X_val = np.array(X_val)
		y_val = np.array(y_val)
		loss, accuracy = model.evaluate(X_val, y_val)
    
		if accuracy > best_params['max']:
			best_params['max'] = accuracy
			best_params['params'] = param_combination
	return best_params


# Function to convert probabilities into pos/neg labels
#
# Arguments:
# probs (numpy.ndarray): A numpy vector containing probability of being positive
#
# Returns:
# pred (numpy.ndarray): A numpy vector containing pos/neg labels such that ith value in probs is mapped to ith value in pred
# 						A value is mapped to pos label if it is >=0.5, neg otherwise
def translate_probs(probs):
	# [YOUR CODE HERE]
	pred = np.repeat('pos', probs.shape[0])
	for i in range(0, len(probs)):
		if probs[i] >= 0.5:
			pred[i] = 'pos'
		else:
			pred[i] = 'neg'
	return pred


# Use the main function to test your code when running it from a terminal
# Sample code is provided to assist with the assignment, it is recommended
# that you do not change the code in main function for this assignment
# You can run the code from termianl as: python3 q3.py
# It should produce the following output and 2 files (q1-train-rep.npy, q1-pred.npy):
#
# $ python3 q1.py
# Best parameters: {'layers': 1, 'units': 8, 'loss': 'binary_crossentropy', 'optimizer': 'adam', 'epochs': 1}

def main():
	# Load dataset
	data = load_data('movie_reviews.csv')

	# Extract list of reviews from the training set
	# Note that since data is already sorted by review IDs, you do not need to sort it again for a subset
	train_data = list(filter(lambda x: x['split'] == 'train', data))
	reviews_train = [r['text'] for r in train_data]

	# Compute the word2vec representation for training set
	X_train = w2v_rep(reviews_train)
	# Save these representations in q1-train-rep.npy for submission
	np.save('q1-train-rep.npy', X_train)

	# Write your code here to extract representations for validation (X_val) and test (X_test) set
	# Also extract labels for training (y_train) and validation (y_val)
	# Use 1 to represent 'pos' label and 0 to represent 'neg' label
	# [YOUR CODE HERE]
	test_data = list(filter(lambda x: x['split'] == 'test', data))
	val_data = list(filter(lambda x: x['split'] == 'val', data))

	X_val = w2v_rep([r['text'] for r in val_data])
	X_test = w2v_rep([r['text'] for r in test_data])
	y_train = [r['label'] for r in train_data]
	y_val = [r['label'] for r in val_data]
	# convert 'pos' to 1 and 'neg' to 1
	y_train = list(map(lambda x: 1 if x == 'pos' else 0, y_train))
	y_val = list(map(lambda x: 1 if x == 'pos' else 0, y_val))

	# Build a feed forward neural network model with build_nn function
	params = {
		'layers': 1,
		'units': 8,
		'loss': 'binary_crossentropy',
		'optimizer': 'adam'
	}
	reset_seeds()
	model = build_nn(params)

	# Function to choose best parameters
	# You should use build_nn function in find_best_params function
	params = {
		'layers': [1, 3],
		'units': [8, 16, 32],
		'loss': ['binary_crossentropy', 'mse'],
		'optimizer': ['sgd', 'adam'],
		'epochs': [1, 5, 10]
	}
	best_params = find_best_params(params, X_train, y_train, X_val, y_val)

	# Save the best parameters in q1-params.csv for submission
	print("Best parameters: {0}".format(best_params['params']))
	# np.save('q1-params.csv', best_params['params'])
	param_list = [["layers", str(best_params['params']['layers'])], 
				["units", str(best_params['params']['units'])], 
				["loss", str(best_params['params']['loss'])],
				["optimizer", str(best_params['params']['optimizer'])],
				["epochs", str(best_params['params']['epochs'])]]
	np.savetxt('q1-params.csv', param_list, delimiter=',', fmt='%s')

	# Build a model with best parameters and fit on the training set
	# reset_seeds function must be called immediately before build_nn and model.fit function
	# Uncomment the following 4 lines to call the necessary functions
	reset_seeds()
	model = build_nn(best_params['params'])
	reset_seeds()
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	model.fit(X_train, y_train, epochs=best_params['params']['epochs'])

	# Use the model to predict labels for the validation set (uncomment the line below)
	pred = model.predict(X_val).flatten()

	# Write code here to evaluate model performance on the validation set
	# You should compute precision, recall, f1, accuracy
	# Save these results in q1-res.csv for submission
	# Can you use translate_probs function to facilitate the conversions before comparison?
	# [YOUR CODE HERE]
	pred_temp = pred
	pred_temp = translate_probs(pred_temp)
	pred = np.zeros(len(pred_temp)).astype(int)
	for i in range(0, len(pred)):
		if pred_temp[i] == 'pos':
			pred[i] = 1
		else:
			pred[i] = 0

	y_val = np.array(y_val)
	p = tf.keras.metrics.Precision()
	p.update_state(y_val, pred)
	precision = p.result().numpy()

	r = tf.keras.metrics.Recall()
	r.update_state(y_val, pred)
	recall = r.result().numpy()

	f1_score = (2*precision*recall) / (precision + recall)

	metrics = [["precision", str(precision)], ["recall", str(recall)], ["f1 score", str(f1_score)], ["accuracy", str(best_params["max"])]]
	np.savetxt('q1-res.csv', metrics, delimiter=',', fmt='%s')
	# Just dummy data to avoid errors
	# pred = np.zeros((10))
	# Use the model to predict labels for the test set (uncomment the line below)
	pred = model.predict(X_test)
	
	# Translate predicted probabilities into pos/neg labels
	pred = translate_probs(pred)
	# Save the results for submission
	np.save('q1-pred.npy', pred)


if __name__ == '__main__':
	main()
