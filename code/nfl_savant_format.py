import os
import csv
import cPickle

import numpy as np
import pandas as pd


# Original columns
cols = ['GameId', 'GameDate', 'Quarter', 'Minute', 'Second', 'OffenseTeam',
'DefenseTeam', 'Down', 'ToGo', 'YardLine', '', 'SeriesFirstDown', '',
'NextScore', 'Description', 'TeamWin', '', '', 'SeasonYear', 'Yards',
'Formation', 'PlayType', 'IsRush', 'IsPass', 'IsIncomplete',
'IsTouchdown', 'PassType', 'IsSack', 'IsChallenge', 'IsChallengeReversed',
'Challenger', 'IsMeasurement', 'IsInterception', 'IsFumble', 'IsPenalty',
'IsTwoPointConversion', 'IsTwoPointConversionSuccessful', 'RushDirection',
'YardLineFixed', 'YardLineDirection', 'IsPenaltyAccepted', 'PenaltyTeam',
'IsNoPlay', 'PenaltyType', 'PenaltyYards']

# Columns after preprocessing
features = ['Seconds remaining in quarter', 'Quarter', 'Down', 'ToGo',
'Yardline', 'OffenseTeam', 'DefenseTeam']
# NB: OffenseTeam and DefenseTeam are one-hots

# Target labels (mutually exclusive)
labels = ['', 'SHORT LEFT PASS', 'SHORT RIGHT PASS', 'SHORT MIDDLE PASS',
'DEEP LEFT PASS', 'DEEP RIGHT PASS', 'DEEP MIDDLE PASS', 'LEFT RUN',
'RIGHT RUN', 'CENTER RUN', 'PUNT', 'FIELD GOAL']


def format_data():
  dataset = os.path.join(os.path.split(__file__)[0], "..", "data", 'nfl_savant_pbp_2014.csv')
  with open(dataset, 'rb') as csvfile:
    pbp = csv.reader(csvfile, delimiter=',')

    count = 0
    iterplays = iter(pbp)
    next(iterplays) #skip titles

    processed_plays = []
    for play in iterplays:
      count += 1
      processed_play = []

      # If down is zero (quarters, timeouts, kickoff, etc), don't include play
      if int(play[7]) == 0:
        continue

      # Determine label from a few options
      label = label_from_options([play[21], play[26], play[37]])
      processed_play.append(label)
      if label == 0:
        print('found 0 label')
        continue

      # Seconds remaining in current period
      processed_play.append(seconds_remaining(int(play[3]), int(play[4])))

      # Quarter
      processed_play.append(int(play[2]))

      # Down
      processed_play.append(int(play[7]))

      # Yards to go
      processed_play.append(int(play[8]))

      # Yard line
      processed_play.append(int(play[9]))

      # Offense team one-hot
      processed_play.extend(one_hot_team(play[5]))

      # Defense team one-hot
      processed_play.extend(one_hot_team(play[6]))

      processed_plays.append(processed_play)
      if count > 100:
        break

    # print(processed_plays)
    return processed_plays

def pickle_data(data):
  print('pickle_data TODO')

def label_from_options(options):
  # 0: play type (punt, field goal)
  # 1: pass type (short l, short r, short m, deep l, deep r, deep m)
  # 2: run type (end l, end r, tackle l, tackle r, guard l, guard r, center)

  target = ''

  if 'PUNT' in options[0] or 'FIELD GOAL' in options[0]:
    target = options[0]

  if options[1] is not '':
    target = options[1] + ' PASS'

  if options[2] is not '':
    if 'RIGHT' in options[2]:
      target = 'RIGHT RUN'
    elif 'LEFT' in options[2]:
      target = 'LEFT RUN'
    elif 'CENTER' in options[2]:
      target = 'CENTER RUN'


  print(options)
  return labels.index(target)

def seconds_remaining(m, s):
  return m * 60 + s

# Teams
teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL',
'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC','MIA', 'MIN', 'NE', 'NO',
'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN',
'WAS']

def one_hot_team(team):
  one_hot = np.zeros(len(teams))

  for i, t in enumerate(teams):
    one_hot[i] = int(t == team)
  
  return one_hot


if __name__ == '__main__':
  formatted_data = format_data()
  pickle_data(formatted_data)
