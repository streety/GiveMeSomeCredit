"""Logistic regression on the Principal Components"""



import numpy as np
import mdp
from sklearn import cross_validation
from sklearn import linear_model

import sys

import load_data
from base import calc_score, split_data
from logistic_base_model import Logistic_Base_Model, prepare_params


class Logistic_Pca_Model(Logistic_Base_Model):

	def load_train_data(self,data):
		data = super(Logistic_Pca_Model, self).load_train_data(data)
		X, y = data

		X = self.normalise_data(X)

		self.pcan = mdp.nodes.PCANode(output_dim=0.99)
		self.pcan.train(X)
		train_data = self.pcan.execute(X)
		return (train_data, y)
	
	def load_data(self, data, cv=True):
		data = super(Logistic_Pca_Model, self).load_data(data, cv)
		X, y = data

		X = self.normalise_data(X)

		cv_test_data = self.pcan.execute(X)
		return (cv_test_data, y)
	
	# def train_model(self, X, y, params):
	# 	self.clf = linear_model.LogisticRegression(C=params['c'], penalty='l2')
	# 	self.clf.fit(X, y)
	# 	return self.clf
	
	# def run_model(self, X):
	# 	probs = self.clf.predict_proba(X)
	# 	return probs[:,1]







if __name__ == '__main__':


	if len(sys.argv) == 1:

		data = load_data.load_data()

		# Set up datasets for cross validation
		rs = cross_validation.ShuffleSplit(150000, n_iterations=3, test_fraction=.30)

		C, train_results, cv_results = prepare_params(0.0000000001, 0.0001, 10)
		
		# Run through the cross validation iterations
		for train_index, cv_index in rs:
			train_data, cv_data = split_data(data[1], train_index, cv_index)
					
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
	





	if '--gen' in sys.argv[1]:
		options = sys.argv[1].split('=')
		c = options[1]
		data = load_data.load_data()

		model = Logistic_Pca_Model()
		train_X, train_y = model.load_train_data(data[1])
		test_X, test_y = model.load_data(data[2], False)

		model.train_model(train_X, train_y, {'c':float(c)})
		test_predictions = model.run_model(test_X)

		for i,v in enumerate(test_predictions):
			print '{0},{1}'.format(i+1,v)


		