import numpy as np

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data


# Train model
def train_svc(data):
  print('Training SVC...')

  model = svm.LinearSVC()
  train_model(model, data)

def train_random_forest(data):
  print('Training random forest...')

  model = RandomForestClassifier()
  train_model(model, data)

def train_model(model, data):
  train_set_x, train_set_y = data[0]
  test_set_x, test_set_y = data[1]

  model.fit(train_set_x, train_set_y)
  test_accuracy(model, test_set_x, test_set_y)

# Accuracy
def test_accuracy(model, test_set_x, test_set_y):
  print('Calculating accuracy...')

  predictions = model.predict(test_set_x)
  score = np.mean(predictions == test_set_y) * 100
  print('Training completed with test accuracy: %.1f%%.' % score)

  # 0, 1, 2, 3
  # RUN, PASS, PUNT, FIELD_GOAL

  labels = {}
  for j in xrange(0, 4):
    labels[j] = {
      'c': 0,
      'i': 0,
      0: 0,
      1: 0,
      2: 0,
      3: 0
    }

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

  text_labels = ['Run', 'Pass', 'Punt', 'Field goal']
  for j in xrange(0, 4):
    c = labels[j]['c']
    i = labels[j]['i']
    t = i + c + 0.
    accuracy = c / t * 100
    print('%s play call accuracy: %.1f%%.' % (text_labels[j], accuracy))

# Run
if __name__ == '__main__':
  data = load_data('formatted_veltman_pbp_normalized_small.pkl')
  train_random_forest(data)
