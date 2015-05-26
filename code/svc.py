import os
import cPickle
import numpy as np
from sklearn import svm

# Data path
data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')

# Train model
def train_model(data):
  train_set_x, train_set_y = data[0]
  test_set_x, test_set_y = data[1]

  print('Training model...')

  clf = svm.SVC()
  clf.fit(train_set_x, train_set_y)

  prediction = clf.predict(test_set_x[1])
  print('prediction: {0}, label: {1}'.format(prediction, test_set_y[1]))

# Load data
def load_data(dataset):
  print('Loading dataset...')

  dataset = os.path.join(data_path, dataset)
  unpickled = cPickle.load(open(dataset, 'rb'))

  y = unpickled[:, 0] # Labels in first col
  x = unpickled[:, 1:] # Game situation data in following cols

  # Divide into 80% training and 20% test data
  n_train = int(0.8 * len(x))
  n_total = len(x)

  train_set_x = x[:n_train, :]
  train_set_y = y[:n_train]

  test_set_x = x[n_train:n_total, :]
  test_set_y = y[n_train:n_total]

  return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


# Run
if __name__ == '__main__':
  data = load_data('formatted_veltman_pbp_small.pkl')
  train_model(data)
