import pandas as pd
from datetime import datetime
import numpy as np
import os
import random
import tensorflow as tf
#from tensorflow.keras import backend as K
import correctness

seed_value = 110

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


import stock_model


#zaciaganie wartosci wig20
w20_data = pd.read_csv("w20.csv")
w20_data["Data"] = pd.to_datetime(w20_data["Data"], format="%Y-%m-%d")

#wyciaganie cen zamkniecia
data = w20_data.filter(['Zamkniecie'])
dataset = data.values

#zbior uczacy - naleza do niego wartosci wig20 z lat 2004-16
train_data = w20_data[(w20_data["Data"] >= datetime(2004,1, 1)) & (w20_data["Data"] < datetime(2018, 1, 1))]
train_data = np.array(train_data["Zamkniecie"])
train_data = np.reshape(train_data, (-1, 1))

#zbior walidacyjny - naleza do niego wartosci wig20 z roku 2017
val_data = w20_data[(w20_data["Data"] >= datetime(2018,1, 1)) & (w20_data["Data"] < datetime(2019, 1, 1))]
val_data = np.array(val_data["Zamkniecie"])
val_data = np.reshape(val_data, (-1, 1))

#wielkosc zbioru testowego
training_data_len = len(train_data)

#tworzenie modelu
md = stock_model.train_1day_model(train_data, 60, 50, 15, 32, 50, 0.1)

#zbior walidacyjny
val_set = stock_model.make_val_set(train_data, val_data, 60, 10)

#walidacja
val_result = stock_model.validate(md, val_set)

#obliczenia tabeli "poprawnosci" predykcji
corr = correctness.create_correctness_matrix(val_result["actuals"], val_result["forecasts"], len(val_result["forecasts"]), [1,2,5,10], [0.01,0.02,0.05,0.1])
correctness.print_correctness(corr)