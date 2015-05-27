import os
import cPickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# Data path
data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')

# Train model
def train_model(data):
  train_set_x, train_set_y = data[0]
  test_set_x, test_set_y = data[1]

  print('Training model...')

  # model = svm.SVC()
  # model.fit(train_set_x, train_set_y)

  model = RandomForestClassifier()
  model.fit(train_set_x, train_set_y)

  test_accuracy(model, test_set_x, test_set_y)

# Accuracy
def test_accuracy(model, test_set_x, test_set_y):
  print('Calculating accuracy...')

  predictions = model.predict(test_set_x)
  score = np.mean(predictions == test_set_y)
  print(score)

  # 0, 1, 2, 3
  # RUN, PASS, PUNT, FIELD_GOAL

  labels = {}
  preds = {}
  for j in xrange(0, 4):
    labels[j] = {
      'c': 0,
      'i': 0,
      0: 0,
      1: 0,
      2: 0,
      3: 0
    }
    preds[j] = 0

  for j in xrange(1, len(test_set_x)):
    x = test_set_x[j]
    y = test_set_y[j]

    pred = model.predict(x)
    correct = pred[0] == y

    if correct:
      labels[y]['c'] += 1
    else:
      labels[y]['i'] += 1

    labels[y][pred[0]] += 1
    preds[pred[0]] += 1

  print(labels)
  print(preds)


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
