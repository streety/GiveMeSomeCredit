import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics

import load_data

def prepare_dataset(data):
	"""Split in x and y, deal with missing values etc"""
	y = []
	X = []

	means = {}
	var = {}
	num_attribs = len(data[0])
	for i in range(2, num_attribs):
		values = []
		for row in data:
			if row[i] != 'NA':
				values.append(float(row[i]))
		valuesa = np.array(values)
		means[i] = valuesa.mean()
		var[i] = valuesa.var()
	
	for row in data:
		y.append(float(row[1]))
		x = []
		for i in range(2, num_attribs):
			if row[i] == 'NA':
				x.extend([-1., 0.])
			else:
				x.extend([1., (float(row[i]) - means[i]) / var[i] ])
		X.append(x)
	return (np.array(X), np.array(y))

def load_data():
	data = load_data.load_data()
	# load_data.basic_stats(data)

	dataset = prepare_dataset(data[1])
	return dataset

def run_model(dataset):
	rs = cross_validation.ShuffleSplit(150000, n_iterations=3, test_fraction=.30)

	C = [0.01]
	train_results = {0.01:[]}
	cv_results = {0.01:[]}
	while C[-1] <= 200:
		C.append(C[-1] * 2)
		train_results[C[-1]] = []
		cv_results[C[-1]] = []

	# print C

	for train_index, cv_index in rs:
		# print train_index[0:10], cv_index[0:10]
		train_data = dataset[0][train_index]
		train_y = dataset[1][train_index]
		cv_data = dataset[0][cv_index]
		cv_y = dataset[1][cv_index]
		# print len(train_data), len(cv_data)
		for c in C:
			clf = linear_model.LogisticRegression(C=c, penalty='l2')
			clf.fit(train_data, train_y)

			train_predictions = clf.predict(train_data)
			train_roc = metrics.roc_curve(train_y, train_predictions)
			train_fpr, train_tpr, train_thresholds = train_roc
			train_auc = metrics.auc(train_fpr, train_tpr)
			train_results[c].append(train_auc)

			cv_predictions = clf.predict(cv_data)
			cv_roc = metrics.roc_curve(cv_y, cv_predictions)
			cv_fpr, cv_tpr, cv_thresholds = cv_roc
			cv_auc = metrics.auc(cv_fpr, cv_tpr)
			cv_results[c].append(cv_auc)


	# print train_results
	# print cv_results

	# for i,v in train_results.iteritems():
	# 	print i, np.array(v).mean()

	# for i,v in cv_results.iteritems():
	# 	print i, np.array(v).mean()

	keys = train_results.keys()
	keys.sort()
	for k in keys:
		print '{0:.3f} : {1:.3f} : {2:.3f}'.format(k,
							np.array(train_results[k]).mean(),
							np.array(cv_results[k]).mean())





if __main__ == '__main__':
	dataset = load_data()
	run_model(dataset)