# Dataset from Noah Veltman (https://github.com/veltman/nflplays)

import os
import csv
import cPickle

import numpy as np
import pandas as pd


# Columns
columns = ['PLAY_ID', 'GAME_ID', 'QTR', 'MIN', 'SEC', 'OFF', 'DEF', 'DOWN',
'YARDS_TO_FIRST', 'YARDS_TO_GOAL', 'DESCRIPTION', 'OFF_SCORE', 'DEF_SCORE',
'SEASON', 'YEAR', 'MONTH', 'DAY', 'HOME_TEAM', 'AWAY_TEAM', 'PLAY_TYPE']

# Features
features = ['QTR', 'SECONDS_REMAINING', 'OFF_TEAM', 'DEF_TEAM', 'DOWN',
'YARDS_TO_FIRST', 'YARDS_TO_GOAL', 'OFF_SCORE', 'DEF_SCORE', 'YEAR', 'MONTH',
'DAY', 'OFF_IS_HOME', 'PLAY_TYPE']

# Labels
labels = ['', 'PASS', 'RUN', 'PUNT', 'FIELD_GOAL']

# Teams
teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC','MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ',
'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS']

# Data path
data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')


def format_data():
  dataset = os.path.join(
    os.path.split(__file__)[0], "..", "data", 'veltman_pbp_2002_2012.csv')

  with open(dataset, 'rb') as csvfile:
    plays = csv.reader(csvfile, delimiter='\t')

    processed_plays = []
    unlabeled_plays = 0

    for i, play in enumerate(plays):
      if i == 0: continue
      processed_play = []

      # If down is zero (quarters, timeouts, kickoff, etc), skip the play
      if int(play[7]) == 0: continue

      # Determine label from a few options
      label = label_from_options(play[19])
      processed_play.append(label)
      if label == 0:
        # print('Unlabeled: {0}\n'.format(play[19]))
        unlabeled_plays += 1
        continue

      # Quarter
      processed_play.append(int(play[2]))

      # Seconds remaining in game
      processed_play.append(seconds_remaining(int(play[3]), int(play[4])))

      # Down
      processed_play.append(int(play[7]))

      # Yards to go
      processed_play.append(int(play[8]))

      # Yard to goal
      processed_play.append(int(play[9]))

      # Offense team one-hot
      off_team = play[5]
      processed_play.extend(one_hot_team(off_team))

      # Defense team one-hot
      processed_play.extend(one_hot_team(play[6]))

      # Offense team score
      processed_play.append(int(play[11]))

      # Defense team score
      processed_play.append(int(play[12]))

      # Year
      processed_play.append(int(play[14]))

      # Month
      processed_play.append(int(play[15]))

      # Day
      processed_play.append(int(play[16]))

      # Offense is home team
      processed_play.append(offense_is_home(off_team, play[17]))

      processed_plays.append(processed_play)
      # if i > 1:
      #   break

    print('Unlabeled: {0}'.format(unlabeled_plays))
    print('Labeled: {0}'.format(len(processed_plays)))
    return processed_plays

def pickle_data(data):
  save_path = os.path.join(data_path, 'formatted_veltman_pbp.pkl')
  print('Saving formatted play by play data to {0}'.format(save_path))

  cPickle.dump(data, open(save_path, 'wb'))

def label_from_options(options):
  # options = options.split('|')
  target = ''

  if 'RUN' in options:
    target = 'RUN'
  elif 'PASS' in options or 'INTERCEPTION' in options:
    target = 'PASS'
  elif 'PUNT' in options:
    target = 'PUNT'
  elif 'FIELD_GOAL' in options:
    target = 'FIELD_GOAL'

  return labels.index(target)

def seconds_remaining(m, s):
  return m * 60 + s

def one_hot_team(team):
  one_hot = np.zeros(len(teams))

  for i, t in enumerate(teams):
    one_hot[i] = int(t == team)
  
  return one_hot

def offense_is_home(off_team, home_team):
  return int(off_team == home_team)


if __name__ == '__main__':
  formatted_data = format_data()
  pickle_data(formatted_data)