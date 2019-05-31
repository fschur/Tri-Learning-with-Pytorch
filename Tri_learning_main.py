import numpy as np
import torch
from Networks import Highway
from TriLearning import TriTraining
import torch.nn as nn
from preprocessing_tri import preprocessing_tri
import csv

# setting seeds
np.random.seed(153)
torch.manual_seed(4623)

# if train_mode=True then then trainingsset is split into 2 sets for hyperparameter tuning
train_mode = True

# getting the data
if train_mode == False:
    (train_x, train_y, scaled_x_un, scaled_test_x) = preprocessing_tri(train=train_mode)
else:
    (train_x, train_y, scaled_x_un, dev_x, dev_y, scaled_test_x) = preprocessing_tri(train=train_mode)

# defining the models
model_1 = nn.Sequential(Highway(size=139, num_layers=5, f=nn.functional.leaky_relu, dropout_rate=0.3)
                      , nn.Linear(139, 10))

model_2 = nn.Sequential(Highway(size=139, num_layers=5, f=nn.functional.leaky_relu, dropout_rate=0.3)
                      , nn.Linear(139, 10))

model_3 = nn.Sequential(Highway(size=139, num_layers=5, f=nn.functional.leaky_relu, dropout_rate=0.3)
                      , nn.Linear(139, 10))

# initializing the tri-net
Trainer = TriTraining([model_1, model_2, model_3])

# training Tri-net
Trainer.fit(train_x, train_y, scaled_x_un)

# if train_mode == False, then safe the predictions in solutions.csv
if train_mode == False:
    result = Trainer.predict(scaled_test_x)

    result = zip(range(30000, 30000 + len(result)), result)
    print(result)
    with open('solutions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Id', 'y'))
        writer.writerows(result)
# if train_mode==False, then calculate the score on the dev_set
else:
    Trainer.score(dev_x, dev_y)