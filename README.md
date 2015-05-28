# nfl-play-prediction
Crudely predicting NFL play call from game situations. [The base dataset](https://github.com/veltman/nflplays) covers the regular and postseason for the years 2002-2012.

----

#### Input
Each play call prediction is based off a game situation that includes the following parameters:
* quarter
* seconds remaining in game
* offensive team
* defensive team
* down
* yards to first down
* yards to goal
* offensive team score
* defensive team score
* year
* month
* day
* home team


#### Labels
Play types are divided into four labels:
* run
* pass
* punt
* field goal

Any other game situations (e.g., conversions or kick-offs) were excluded from training.


#### Models
Two models were trained on the play by play data: an SVC and a random forest. You can run either by modifying and running code/models.py to suit your preference.


#### Game
You can test your own ability as a coach by running code/guess.py -- a text-based play calling game. You are presented the game situation and you must make the correct call.


----
#### Roadmap
A trove of relevant information is likely stored in the preceding sequences of plays. Better test accuracy could probably be achieved by ordering the play by play data by series and accounting for the preceding play calls and outcomes. It would be especially interesting to tackle this problem with an RNN. Even without play sequences, using a [vanilla neural net](https://github.com/cgyulay/theano-nn) might yield better training accuracy.
