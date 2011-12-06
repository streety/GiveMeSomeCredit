"""PCA"""



import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mdp


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


def load_dataset():
	data = load_data.load_data()
	# load_data.basic_stats(data)

	dataset = prepare_dataset(data[1])
	return dataset


def run_pca(dataset):
	X = dataset[0]

	pcan = mdp.nodes.PCANode(output_dim=0.97)
	pcar = pcan.execute(X)
	print pcan.d

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(pcar[:,0], pcar[:,1], 'k.')
	plt.show()


if __name__ == '__main__':
	dataset = load_dataset()
	run_pca(dataset)