import numpy as np

from load_data import load_data
from veltman_format import teams

class PlayCallGame():
  def __init__(self):
    self.guesses = []
    self.answers = []
    self.start()

  def start(self):
    data = load_data('formatted_veltman_pbp_small.pkl', False)
    self.train_set_x, self.train_set_y = data[0]
    self.test_set_x, self.test_set_y = data[1]

    # Opening prompt
    print('\nTry your luck as an NFL coach! Guess the play call based on each '
      '(admittedly simple) game situation.')
    inpt = raw_input('Type \'q\' at any time to stop. '
      'Press enter to begin...\n')

    n_correct = 0
    n_incorrect = 0

    # Game loop
    if inpt != 'q':
      response = ''
      while response != 'q':
        response, answer = self.ask_question()
        if response == 'q':
          self.end_game(n_correct, n_incorrect)
          continue

        response = int(response) - 1

        if response == answer:
          print('Good call, coach!\n')
          n_correct += 1
        else:
          action = self.format_action(answer)
          print('Whoops, that\'s not what your NFL counterpart decided.'
            ' He {0}.\n'.format(action))
          n_incorrect += 1
    else:
      self.end_game(n_correct, n_incorrect)

  def ask_question(self):
    row, answer = self.select_random()
    off_team, def_team = self.extract_teams(row)

    score = self.format_score(off_team, def_team, int(row[69]), int(row[70]))
    situation = 'You are coaching {0}. {1} with {2} in the {3}.'.format(
      off_team, score, self.format_time(row), self.format_quarter(row))
    position = self.format_position(row, def_team)
    question = situation + ' ' + position + (' Will you [1] run the ball, [2] '
      'pass, [3] punt, or [4] kick a field goal? ')

    return raw_input(question), answer

  def format_score(self, off_team, def_team, off_score, def_score):
    order = ()
    if off_score > def_score:
      order = (off_team, def_team, off_score, def_score)
    elif off_score < def_score:
      order = (def_team, off_team, def_score, off_score)
    else:
      return '{0} and {1} are tied at {2}'.format(off_team, def_team, off_score)

    return '{0} leads {1} {2}-{3}'.format(*order)

  def format_time(self, row):
    quarter = row[0]
    seconds = row[1] - (4 - quarter) * 60 * 15
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return '{0}:{1:02d}'.format(minutes, seconds)

  def format_quarter(self, row):
    quarter = int(row[0])
    return '{0}{1} quarter'.format(quarter, self.suffix(quarter))

  def format_position(self, row, def_team):
    down = int(row[2])
    togo = int(row[3])

    yardline = int(row[4])
    if yardline > 50:
      yardline -= 50
      yardline_str = 'your own {0} yard line'.format(yardline)
    else:
      yardline_str = '{0}\'s {1} yard line'.format(def_team, yardline)
      if yardline <= 10: togo = 'goal'

    return 'It\'s {0}{1} and {2} on {3}.'.format(down,
      self.suffix(down), togo, yardline_str)

  def format_action(self, answer):
    actions = ['ran the ball', 'threw the ball', 'punted the ball', 'kicked a '
      'field goal']

    return actions[answer]

  def select_random(self):
    row = np.random.randint(1, len(self.test_set_x))
    return self.test_set_x[row], self.test_set_y[row]

  def extract_teams(self, row):
    off_one_hot = row[5:37]
    def_one_hot = row[37:69]

    off_team = teams[int(np.argmax(off_one_hot))]
    def_team = teams[int(np.argmax(def_one_hot))]
    return off_team, def_team

  def suffix(self, index):
    suffixes = ['st', 'nd', 'rd', 'th']
    return suffixes[index - 1]

  def end_game(self, n_correct, n_incorrect):
    n_total = n_correct + n_incorrect
    accuracy = n_correct / (n_total + 0.) * 100 if n_total > 0 else 0.

    print('\nThanks for playing, coach. You correctly guessed {0} out of {1} '
      'play calls for an accuracy of {2:.1f}%.'.format(n_correct,
      n_total, accuracy))


# Run
if __name__ == '__main__':
  PlayCallGame()
  