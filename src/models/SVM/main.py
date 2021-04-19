import sys
import os
sys.path.append('./imports')
sys.path.append('./config')
sys.path.append('./database')

import json

from utils.imports import *
from imports import model

with open('./config/config.json', 'r') as json_file:
    file = json.load(json_file)

dataframe = pd.read_csv('database/WEGE3/WEGE3_M15_data.csv')

model = model.SVM()

model.build_svm()
model.split_train_test_data(dataframe, size=200)
model.train_model()
model.print_results()