import csv
import numpy as np
import matplotlib.pyplot as plt

def load_data(train_file='cs-training.csv', test_file='cs-test.csv'):
	"""Load data"""
	train_f = open(train_file, 'r')
	train_reader = csv.reader(train_f, delimiter=",", quotechar='"')
	train_data = []
	for row in train_reader:
		train_data.append(row)
	
	train_f.close()

	test_f = open(test_file, 'r')
	test_reader = csv.reader(test_f, delimiter=",", quotechar='"')
	test_data = []
	for row in test_reader:
		test_data.append(row)
	
	test_f.close()

	headers = train_data[0]

	return (headers, train_data[1:], test_data[1:])

def basic_stats(data):
	"""Construct basic stats on the data"""
	headers, training, testing = data
	print '{0} records in the training set'.format(len(training))
	print '{0} records in the test set'.format(len(testing))

	num_attribs = len(headers)
	for i in range(num_attribs):
		train_values = []
		train_missing = 0
		for row in training:
			if row[i] == 'NA':
				train_missing += 1
			elif row[i]:
				train_values.append(float(row[i]))
						
		
		test_values = []
		test_missing = 0
		for row in testing:
			if row[i] == 'NA':
				test_missing += 1
			elif row[i]:
				test_values.append(float(row[i]))
						
		
		print '{0:<37}: {1:.2f} ({2}) : {3:.2f} ({4})'.format(headers[i], 
											float(np.array(train_values).mean()),
											train_missing,
											float(np.array(test_values).mean()),
											test_missing,)

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
	return (X, y)


