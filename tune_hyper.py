import os
import random
import pandas as pd
from datetime import datetime
import numpy as np
import json
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras import backend as K

seed_value = 110

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

import stock_model

if os.path.exists("models.csv"):
  os.remove("models.csv")

try:
    model_summary = pd.read_csv("models.csv")
except FileNotFoundError:
    model_summary = pd.DataFrame(columns=["model_idx", "lstm_units", "dense_units",
                              "drop", "batch", "epochs", "mean", "sd"])

def val_get_stats(val_result):
    errors = []
    for i in range(len(val_result["actuals"])):
        errors.append(np.mean(np.abs((val_result["actuals"][i] - val_result["forecasts"][i]))/val_result["actuals"][i]))
    return {"mean": np.mean(errors), "sd": np.std(errors)}

#zaciaganie wartosci wig20
w20_data = pd.read_csv("csv/w20.csv")
w20_data["Data"] = pd.to_datetime(w20_data["Data"], format="%Y-%m-%d")

#wyciaganie cen zamkniecia
data = w20_data.filter(['Zamkniecie'])
dataset = data.values

#zbior uczacy - naleza do niego wartosci wig20 z lat 2004-16
train_data = w20_data[(w20_data["Data"] >= datetime(2004,1, 1)) & (w20_data["Data"] < datetime(2017, 1, 1))]
train_data = np.array(train_data["Zamkniecie"])
train_data = np.reshape(train_data, (-1, 1))

#zbior walidacyjny - naleza do niego wartosci wig20 z roku 2017
val_data = w20_data[(w20_data["Data"] >= datetime(2017,1, 1)) & (w20_data["Data"] < datetime(2018, 1, 1))]
val_data = np.array(val_data["Zamkniecie"])
val_data = np.reshape(val_data, (-1, 1))

with open("hyper.json", "r") as input_file:
    params = json.load(input_file)

start_idx = 0
end_idx = 500

for i in range(start_idx, end_idx):
    lstm_units = params[i]["lstm_units"]
    print(i)

    try:
        md = {"model": None, "lag": 60}
        md["model"] = models.load_model("tmp_md/" + str(i) + ".h5")
    except Exception as e:
        raise e
        print(e)
        md = stock_model.train_1day_model(train_data, 60, params[i]["lstm_units"], params[i]["dense_units"],
                                            params[i]["bat"], params[i]["epochs"], drop_rate=params[i]["drop"])
        md["model"].save("tmp_md/" + str(i) + ".h5")  # DEBUG


    val_set = stock_model.make_val_set(train_data, val_data, 60, 10)
    val_result = stock_model.validate(md, val_set)
    mean_sd = val_get_stats(val_result)
    df_row = {"model_idx": i, "lstm_units": params[i]["lstm_units"], "dense_units": params[i]["dense_units"],
              "drop": params[i]["drop"], "batch": params[i]["bat"], "epochs": params[i]["epochs"],
              "mean": mean_sd["mean"], "sd": mean_sd["sd"]}
    model_summary = model_summary.append(df_row, ignore_index=True)
    model_summary.to_csv("models.csv", sep=';')

print("Summary saved to file models.csv")