from sklearn import linear_model
from base import Base_Model

class Logistic_Base_Model(Base_Model):

	def load_train_data(self,data):
		return super(Logistic_Base_Model, self).load_train_data(data)
	
	def load_data(self, data, cv=True):
		return super(Logistic_Base_Model, self).load_data(data, cv)

	def train_model(self, X, y, params):
		self.clf = linear_model.LogisticRegression(C=params['c'], penalty='l2')
		self.clf.fit(X, y)
		return self.clf
	
	def run_model(self, X):
		probs = self.clf.predict_proba(X)
		return probs[:,1]






def prepare_params(start_c, limit, multiplier):
	C = [start_c]
	train_results = {start_c:[]}
	cv_results = {start_c:[]}
	while C[-1] < limit:
		C.append(C[-1] * multiplier)
		train_results[C[-1]] = []
		cv_results[C[-1]] = []
	return (C, train_results, cv_results)