"""Logistic regression"""



import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics

import load_data

def prepare_dataset(data):
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

def expand_dataset(dataset):
	""" Expand the dataset by adding 2nd polynomial of the 
	    features """
	 
	X, y = dataset

	# Take every other column to miss out value missing 
	# indicating columns
	cols = range(1,X.shape[1],2)
	for i in cols:
		for j in cols:
			new_feature = X[:,i] * X[:,j]
			new_feature = np.atleast_2d(new_feature)
			X = np.concatenate((X, new_feature.T), axis=1)
		
	# Normalise for median and interquartile range
	for i in range(20, X.shape[1]):
		interquartile = stats.scoreatpercentile(X[:,i], 75) - \
							stats.scoreatpercentile(X[:,i], 25)
		X[:,i] = X[:,i] - np.median(X[:,i])
		if interquartile != 0:
			X[:,i] = X[:,i] / interquartile
		
	return (X, y)

def load_dataset():
	data = load_data.load_data()
	# load_data.basic_stats(data)

	dataset = prepare_dataset(data[1])
	return dataset

def run_model(dataset):
	# Set up datasets for cross validation
	rs = cross_validation.ShuffleSplit(150000, n_iterations=3, test_fraction=.30)

	# Set up C values to test
	start_c = 0.00001
	multiplier = 4
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
		train_data = dataset[0][train_index]
		train_y = dataset[1][train_index]
		cv_data = dataset[0][cv_index]
		cv_y = dataset[1][cv_index]

		# Run through the C values to test
		for c in C:
			print c
			clf = linear_model.LogisticRegression(C=c, penalty='l2')
			clf.fit(train_data, train_y)

			train_predictions = clf.predict_proba(train_data)
			train_roc = metrics.roc_curve(train_y, train_predictions[:,1])
			train_fpr, train_tpr, train_thresholds = train_roc
			train_auc = metrics.auc(train_fpr, train_tpr)
			train_results[c].append(train_auc)

			cv_predictions = clf.predict_proba(cv_data)
			cv_roc = metrics.roc_curve(cv_y, cv_predictions[:,1])
			cv_fpr, cv_tpr, cv_thresholds = cv_roc
			cv_auc = metrics.auc(cv_fpr, cv_tpr)
			cv_results[c].append(cv_auc)
	
	# Display results
	keys = train_results.keys()
	keys.sort()
	for k in keys:
		print '{0:.5f} : {1:.4f} : {2:.4f}'.format(k,
							np.array(train_results[k]).mean(),
							np.array(cv_results[k]).mean())





if __name__ == '__main__':
	dataset = load_dataset()
	dataset = expand_dataset(dataset)
	run_model(dataset)