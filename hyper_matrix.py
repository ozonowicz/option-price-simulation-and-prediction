import random
import json

random.seed(200)

mtx = []
for lstm_units in [10, 20, 30, 40, 50]:
    for dense_units in [5, 10, 15, 20, 25]:
        for drop in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for bat in [16, 32, 64, 128]:
                for epochs in [10, 20, 30, 40, 50]:
                    mtx.append({"lstm_units": lstm_units, "dense_units": dense_units, "drop": drop,
                                "bat": bat, "epochs": epochs})
random_grid = random.choices(mtx, k=500)

with open("hyper.json", "w") as outfile:
    json.dump(random_grid, outfile)