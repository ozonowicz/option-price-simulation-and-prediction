import pandas as pd
from datetime import datetime
import numpy as np
import gbm
import black_scholes

def is_outlier(s):
    lower_limit = s.quantile(0.05)
    upper_limit = s.quantile(0.95)
    return ~s.between(lower_limit, upper_limit)


w20_data = pd.read_csv("csv/w20.csv")
w20_data["Data"] = pd.to_datetime(w20_data["Data"], format="%Y-%m-%d")

option_prices = pd.read_csv("csv/option_prices.csv")

option_prices["start_date"] = pd.to_datetime(option_prices["start_date"], format="%d.%m.%Y")
option_prices["end_date"] = pd.to_datetime(option_prices["end_date"], format = "%d.%m.%Y")
option_prices["date"] = pd.to_datetime(option_prices["date"], format="%d.%m.%Y")

wimean = pd.read_csv("csv/wimean_adjusted.csv")
wimean["date"] = pd.to_datetime(wimean["date"], format="%Y-%m-%d")

sigma_frame = pd.DataFrame([], columns=["date", "spot_price", "sigma60"])
for i in range(2723, len(w20_data)):
    prices = w20_data.loc[(i-60):i, "Zamkniecie"].values
    sigma = gbm.train_gbm(prices)["sigma"]
    sigma_frame = sigma_frame.append({"date": w20_data.loc[i, "Data"],
                                      "spot_price": w20_data.loc[i, "Zamkniecie"],
                                      "sigma60": sigma}, ignore_index=True)


option_prices = option_prices.join(sigma_frame.set_index("date"), on="date")

wimean_periods = [14, 30, 90, 180, 270, 365]

option_prices = option_prices.join(wimean.set_index("date"), on="date")
option_prices["r"] = option_prices.apply(
    lambda row: np.interp(row.time_to_maturity, wimean_periods,
                          [row.wimean_2w, row.wimean_1m, row.wimean_3m, row.wimean_6m, row.wimean_9m, row.wimean_1y]),
    axis=1)

option_prices["bs_price"] = option_prices.apply(
    lambda row: max(0.01, black_scholes.euro_vanilla(row.spot_price, row.strike, row.time_to_maturity/254, row.r, row.sigma60, row.type)),
    axis=1
)

option_prices["err"] = np.abs(option_prices["bs_price"] - option_prices["price"])/ option_prices["price"]

option_prices["relative_strike"] = (option_prices["strike"]/option_prices["spot_price"])

option_prices["year"] = option_prices["date"].dt.year

option_prices = option_prices[(option_prices["date"] > datetime(2004,1,1)) & (option_prices["date"] < datetime(2019,1,1))]

def make_summary(col_name, cut_bins):
    if cut_bins:
        range_name = col_name + "_range"
        cuts = [min(option_prices[col_name])] + cut_bins + [max(option_prices[col_name])]
        option_prices[range_name] = pd.cut(option_prices[col_name], cuts, include_lowest=True)
    else:
        range_name = col_name

    grouped = option_prices[~option_prices.groupby(range_name)["err"].apply(is_outlier)].groupby(range_name)["err"]
    summary = pd.DataFrame(grouped.mean())

    x = option_prices.groupby(range_name).size().reset_index(name='count')
    summary = summary.join(x.set_index(range_name), on=range_name)

    for thres in [0.05, 0.1, 0.2, 0.5]:
        thres_col = "C[" + str(thres) + "]"
        x = option_prices[option_prices["err"] < thres].groupby(range_name).size().reset_index(name=thres_col)
        summary = summary.join(x.set_index(range_name), on=range_name)
        summary[thres_col] = summary[thres_col]/summary["count"]
    return summary

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(make_summary("volume", [1, 10, 100]))
print(make_summary("time_to_maturity", [10, 30, 50, 100]))
print(make_summary("price", [1, 10, 50, 100]))
print(make_summary("sigma60", [0.15, 0.20, 0.30]))
print(make_summary("relative_strike", [0.8, 0.9, 1, 1.1, 1.2]))
print(make_summary("year", None))
print(make_summary("type", None))
print(make_summary("r", [0.015, 0.02, 0.03, 0.04, 0.05]))
