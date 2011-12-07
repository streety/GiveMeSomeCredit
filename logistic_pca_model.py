"""Logistic regression on the Principal Components"""



import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mdp

from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics

import sys

import load_data

def prepare_dataset(data, test=False):
	"""Split in x and y, deal with missing values etc"""
	y = []
	X = []

	# Calculate median and interquartile range values
	# for each column.
	# Used later for normalisation.
	median = {}
	interquartile = {}
	num_attribs = len(data[0])
	for i in range(2, num_attribs):
		values = []
		for row in data:
			if row[i] != 'NA':
				values.append(float(row[i]))
		valuesa = np.array(values)
		median[i] = np.median(valuesa)
		interquartile[i] = stats.scoreatpercentile(valuesa, 75) - \
							stats.scoreatpercentile(valuesa, 25)
	
	
	# Normalise values and indicate whether any value is 
	# missing.  This is done by inserted extra columns with
	# either 1. if value present or -1. if value missing
	# Missing values are filled in with 0. - equivalent to a
	# median value.
	for row in data:
		if not test:
			y.append(float(row[1]))
		x = []
		for i in range(2, num_attribs):
			if row[i] == 'NA':
				x.extend([-1., 0.])
				# x.extend([0.])
			elif (interquartile[i] > 0) and (abs(float(row[i]) - median[i]) / interquartile[i]) > 4:
				x.extend([-1., 0.])
				# x.extend([0.])
			else:
				if interquartile[i] == 0:
					value = (float(row[i]) - median[i])
				else:
					value = (float(row[i]) - median[i]) / interquartile[i]
				x.extend([1., value])
				# x.extend([value])
		
		X.append(x)

	return (np.array(X), np.array(y))


def load_dataset():
	data = load_data.load_data()
	# load_data.basic_stats(data)

	dataset = prepare_dataset(data[1])
	return dataset

def normalise_data(data):
	cols = range(0, data.shape[1])
	for i in cols:
		median = np.median(data[:,i])
		interquartile = stats.scoreatpercentile(data[:,i], 75) - \
							stats.scoreatpercentile(data[:,i], 25)
		data[:,i] = (data[:,i] - median) / interquartile
		return data

def calc_score(predictions, actual):
	roc = metrics.roc_curve(actual, predictions)
	fpr, tpr, thresholds = roc
	auc = metrics.auc(fpr, tpr)
	return auc

def generate_model(c, train_data, train_y):
	clf = linear_model.LogisticRegression(C=c, penalty='l2')
	clf.fit(train_data, train_y)
	return clf

def prepare_X_data(train_data, cv_data):
	pcan = mdp.nodes.PCANode(output_dim=0.99)
	pcan.train(train_data)
	train_data = pcan.execute(train_data)
	cv_data = pcan.execute(cv_data)
	train_data = normalise_data(train_data)
	cv_data = normalise_data(cv_data)
	return (train_data, cv_data)

def run_pca_model(dataset):
	X, Y = dataset

	# Set up datasets for cross validation
	rs = cross_validation.ShuffleSplit(150000, n_iterations=4, test_fraction=.30)

	# Set up C values to test
	start_c = 0.0000001
	multiplier = 10
	limit = 1000
	C = [start_c]
	train_results = {start_c:[]}
	cv_results = {start_c:[]}
	while C[-1] <= limit:
		C.append(C[-1] * multiplier)
		train_results[C[-1]] = []
		cv_results[C[-1]] = []
	
	# Run through the cross validation iterations
	for train_index, cv_index in rs:
		train_data = X[train_index]
		train_y = Y[train_index]
		cv_data = X[cv_index]
		cv_y = Y[cv_index]

		print train_data.shape, cv_data.shape

		train_data, cv_data = prepare_X_data(train_data, cv_data)
		
		print train_data.shape, cv_data.shape

		# Run through the C values to test
		for c in C:
			print c
			clf = generate_model(c, train_data, train_y)
			
			train_predictions = clf.predict_proba(train_data)
			train_results[c].append(calc_score(train_predictions[:,1], train_y))

			cv_predictions = clf.predict_proba(cv_data)
			cv_results[c].append(calc_score(cv_predictions[:,1], cv_y))
			
	
	# Display results
	keys = train_results.keys()
	keys.sort()
	for k in keys:
		print '{0:.2e} : {1:.4f} : {2:.4f}'.format(k,
							np.array(train_results[k]).mean(),
							np.array(cv_results[k]).mean())


if __name__ == '__main__':
	if '--gen' in sys.argv[1]:
		options = sys.argv[1].split('=')
		c = options[1]
		data = load_data.load_data()
		train_dataset = prepare_dataset(data[1])
		train_dataset_X, train_dataset_Y = train_dataset

		test_dataset = prepare_dataset(data[2], test=True)
		test_dataset_X = test_dataset[0]

		train_dataset_X, test_dataset_X = prepare_X_data(train_dataset_X, test_dataset_X)
		clf = generate_model(float(c), train_dataset_X, train_dataset_Y)
			
		test_predictions = clf.predict_proba(test_dataset_X)

		for i,v in enumerate(test_predictions[:,1]):
			print '{0},{1}'.format(i+1,v)
	else:
		dataset = load_dataset()
		run_pca_model(dataset)