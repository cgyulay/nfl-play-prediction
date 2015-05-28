import os
import cPickle

import numpy as np

# Data path
data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')

# Load data
def load_data(dataset='formatted_veltman_pbp_normalized_small.pkl',
  verbose=True):

  if verbose: print('Loading dataset...')

  dataset = os.path.join(data_path, dataset)
  unpickled = cPickle.load(open(dataset, 'rb'))

  y = unpickled[:, 0] # Labels in first col
  x = unpickled[:, 1:] # Game situation data in following cols
  y = np.asarray(y, dtype=np.int32)

  # Divide into 80% training and 20% test data
  n_train = int(0.8 * len(x))
  n_total = len(x)

  train_set_x = x[:n_train, :]
  train_set_y = y[:n_train]

  test_set_x = x[n_train:n_total, :]
  test_set_y = y[n_train:n_total]

  return [(train_set_x, train_set_y), (test_set_x, test_set_y)]