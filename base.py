
import numpy as np
from scipy import stats
from sklearn import metrics




class Base_Model(object):
	median = {}
	interquartile = {}

	def load_train_data(self, data):

		X = []
		y = []

		median = {}
		interquartile = {}
		num_attribs = len(data[0])
		for i in range(2, num_attribs):
			values = []
			for row in data:
				if row[i] != 'NA':
					values.append(float(row[i]))
			valuesa = np.array(values)
			self.median[i] = np.median(valuesa)
			self.interquartile[i] = stats.scoreatpercentile(valuesa, 75) - \
								stats.scoreatpercentile(valuesa, 25)
		
		for row in data:
			y.append(float(row[1]))
			x = []
			for i in range(2, num_attribs):
				if row[i] == 'NA':
					x.extend([-1., self.median[i]])
				elif (self.interquartile[i] > 0) and (abs(float(row[i]) - self.median[i]) / self.interquartile[i]) > 4:
					x.extend([-1., self.median[i]])
				else:
					x.extend([1., float(row[i])])
			
			X.append(x)

		return (np.array(X), np.array(y))
	
	def load_data(self, data, cv=True):
		X = []
		y = []
		num_attribs = len(data[0])

		for row in data:
			if cv:
				y.append(float(row[1]))
			
			x = []
			for i in range(2, num_attribs):
				if row[i] == 'NA':
					x.extend([-1., self.median[i]])
				elif (self.interquartile[i] > 0) and (abs(float(row[i]) - self.median[i]) / self.interquartile[i]) > 4:
					x.extend([-1., self.median[i]])
				else:
					x.extend([1., float(row[i])])
			
			X.append(x)

		return (np.array(X), np.array(y))
	
	def normalise_data(self, data):
		# This needs to be changed
		# must use median and interquartile ranges from 
		# training data only
		cols = range(0, data.shape[1])
		for i in cols:
			median = np.median(data[:,i])
			interquartile = stats.scoreatpercentile(data[:,i], 75) - \
								stats.scoreatpercentile(data[:,i], 25)
			if interquartile == 0:
				divisor = 1
			else:
				divisor = interquartile
			
			data[:,i] = (data[:,i] - median) / divisor
		return data
	
	def train_model(self, X, y, params):
		print 'base - must override'
	
	def run_model(self, X):
		print 'base - must override'










def calc_score(predictions, actual):
	roc = metrics.roc_curve(actual, predictions)
	fpr, tpr, thresholds = roc
	auc = metrics.auc(fpr, tpr)
	return auc

def split_data(data, train_index, cv_index):
	train_data = []
	cv_data = []
	for row, train_include, cv_include in zip(data, train_index, cv_index):
		if train_include:
			train_data.append(row)
		
		if cv_include:
			cv_data.append(row)
	return(train_data, cv_data)