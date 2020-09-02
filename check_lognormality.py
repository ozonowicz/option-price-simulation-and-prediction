import pandas as pd
from datetime import datetime
import gbm

w20_data = pd.read_csv("w20.csv")
w20_data["Data"] = pd.to_datetime(w20_data["Data"], format="%Y-%m-%d")
w20_data = w20_data[(w20_data["Data"] >= datetime(2004, 1, 1)) & (w20_data["Data"] < datetime(2019, 1, 1))]
vals = w20_data["Zamkniecie"].values

def lognorm_periods(period_len):
    result = {"periods": 0, "rejected": 0, "percentage": 0}
    for i in range(period_len, len(vals)):
        prices = vals[(i-period_len):i]
        test = gbm.gbm_test(prices, "shapiro", 0.05)
        if test:
            result["rejected"] = result["rejected"] + 1
        result["periods"] = result["periods"] + 1
    result["percentage"] = result["rejected"] / result["periods"]
    return result

print(lognorm_periods(7))
print(lognorm_periods(14))
print(lognorm_periods(30))
print(lognorm_periods(60))
print(lognorm_periods(90))
print(lognorm_periods(180))
print(lognorm_periods(365))

