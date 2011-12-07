"""Logistic regression on the Principal Components"""



import numpy as np
import mdp
from sklearn import cross_validation
from sklearn import linear_model

import sys

import load_data
from base import *


class Logistic_Pca_Model(Base_Model):

	def load_train_data(self,data):
		data = super(Logistic_Pca_Model, self).load_train_data(data)
		X, y = data

		X = self.normalise_data(X)

		self.pcan = mdp.nodes.PCANode(output_dim=0.99)
		self.pcan.train(X)
		train_data = self.pcan.execute(X)
		return (train_data, y)
	
	def load_data(self, data, cv=True):
		data = super(Logistic_Pca_Model, self).load_data(data)
		X, y = data

		X = self.normalise_data(X)

		cv_test_data = self.pcan.execute(X)
		return (cv_test_data, y)
	
	def train_model(self, X, y, params):
		self.clf = linear_model.LogisticRegression(C=params['c'], penalty='l2')
		self.clf.fit(X, y)
		return self.clf
	
	def run_model(self, X):
		probs = self.clf.predict_proba(X)
		return probs[:,1]







if __name__ == '__main__':

	data = load_data.load_data()

	# Set up datasets for cross validation
	rs = cross_validation.ShuffleSplit(150000, n_iterations=4, test_fraction=.30)

	# Set up C values to test
	start_c = 0.0000000001
	multiplier = 10
	limit = 10
	C = [start_c]
	train_results = {start_c:[]}
	cv_results = {start_c:[]}
	while C[-1] <= limit:
		C.append(C[-1] * multiplier)
		train_results[C[-1]] = []
		cv_results[C[-1]] = []
	
	# Run through the cross validation iterations
	for train_index, cv_index in rs:
		# train_data, cv_data = split_data(data[1], train_index, cv_index)
		train_data = []
		cv_data = []
		for row, train_include, cv_include in zip(data[1], train_index, cv_index):
			if train_include:
				train_data.append(row)
			
			if cv_include:
				cv_data.append(row)
		
		model = Logistic_Pca_Model()
		train_X, train_y = model.load_train_data(train_data)
		cv_X, cv_y = model.load_data(cv_data)
		for c in C:
			print c
			model.train_model(train_X, train_y, {'c':c})
			train_probs = model.run_model(train_X)
			train_results[c].append(calc_score(train_probs, train_y))
			cv_probs = model.run_model(cv_X)
			cv_results[c].append(calc_score(cv_probs, cv_y))
	
	# Display results
	keys = train_results.keys()
	keys.sort()
	for k in keys:
		print '{0:.2e} : {1:.4f} : {2:.4f}'.format(k,
							np.array(train_results[k]).mean(),
							np.array(cv_results[k]).mean())



	# if '--gen' in sys.argv[1]:
	# 	options = sys.argv[1].split('=')
	# 	c = options[1]
	# 	data = load_data.load_data()
	# 	train_dataset = prepare_dataset(data[1])
	# 	train_dataset_X, train_dataset_Y = train_dataset

	# 	test_dataset = prepare_dataset(data[2], test=True)
	# 	test_dataset_X = test_dataset[0]

	# 	train_dataset_X, test_dataset_X = prepare_X_data(train_dataset_X, test_dataset_X)
	# 	clf = generate_model(float(c), train_dataset_X, train_dataset_Y)
			
	# 	test_predictions = clf.predict_proba(test_dataset_X)

	# 	for i,v in enumerate(test_predictions[:,1]):
	# 		print '{0},{1}'.format(i+1,v)
	# else:
	# 	dataset = load_dataset()
	# 	run_pca_model(dataset)