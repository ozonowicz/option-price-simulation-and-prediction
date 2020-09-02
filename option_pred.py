import pandas as pd
import os
import random
import tensorflow as tf
from datetime import datetime
import numpy as np
import correctness

seed_value = 110

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

import stock_model
import option_model

#_____________________________________
# ta czesc jest identyczna jak w pliku nn_stock_pred.py

#zaciaganie wartosci wig20
w20_data = pd.read_csv("csv/w20.csv")
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

#tworzenie modelu - ceny akcji
md = stock_model.train_1day_model(train_data, 60, 50, 15, 32, 50, 0.1)
#md = stock_model.train_1day_model(train_data, 60, 10, 10, 32, 10, 0.1)

#______________________________________________________

#wyciaganie cen opcji
option_prices = pd.read_csv("csv/option_prices.csv")
option_prices["start_date"] = pd.to_datetime(option_prices["start_date"], format="%d.%m.%Y")
option_prices["end_date"] = pd.to_datetime(option_prices["end_date"], format = "%d.%m.%Y")
option_prices["date"] = pd.to_datetime(option_prices["date"], format="%d.%m.%Y")

#stopy procentowe
wimean = pd.read_csv("csv/wimean_adjusted.csv")
wimean["date"] = pd.to_datetime(wimean["date"], format="%Y-%m-%d")

pred_days = 20

#tagi opcji notowanych w roku 2017, o czasie trwania powyzej 50 dni
tags = option_prices[(option_prices["start_date"] >= datetime(2018, 1,1)) & \
                     (option_prices["start_date"] < datetime(2019, 1,1)) & \
                     (option_prices["expiry_year"] == 2018) & \
                     (option_prices["duration_trade_days"] >= pred_days)]["tag"].unique()

pred_10 = option_prices[(option_prices["start_date"] >= datetime(2018, 1,1)) & \
                        (option_prices["start_date"] < datetime(2019, 1,1)) & \
                        (option_prices["expiry_year"] == 2018) & \
                        (option_prices["duration_trade_days"] >= pred_days + 10) & \
                        (option_prices["time_to_maturity"] > pred_days) & \
                        (option_prices["time_to_maturity"] <= pred_days + 10)]
pred10_vol = pd.DataFrame(pred_10.groupby("tag", as_index=False)["volume"].sum())
print(pred10_vol[pred10_vol["volume"] > 0])
tags = pred10_vol[pred10_vol["volume"] > 0]["tag"].unique()
print(tags)


mtx = option_model.option_pred_matrix(md, option_prices, w20_data, wimean, tags, pred_days, 10, 60, 254)
corr = correctness.create_correctness_matrix(mtx["actuals"], mtx["forecasts"], len(mtx["forecasts"]), [1,2,5,10], [0.05,0.1,0.2,0.5])
correctness.print_correctness(corr)