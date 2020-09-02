import black_scholes
import gbm
import numpy as np
import stock_model

#predykcja cen opcji na podstawie modelu cen akcji (podobnego jak w nn_stock_pred) i stop procentowych
#model - model cen akcji
#option_prices - ramka danych z cenami opcji
#stock_prices - ramka danych z cenami akcji / wartosciami indeksu
#rates - ramka danych z wartosciami stop procentowych wimean
#tag - identyfikator aktywa
#time_start - w jakim momencie (dni do wygasniecia) trwania opcji rozpoczynac predykcje
#pred_steps - liczba dni predykcji
#lag - parametr z modelu nn_stock_pred - na podstawie ilu wcześniejszych obserwacji przewiduje się ceny akcji
#days_in_year - ile notowań jest w roku - zazwyczaj przyjmuje się stałą wartość np. 250 lub 254

#wartosc wyjsciowa - slownik z dwoma wpisami
# "actual": wektor z prawdziwymi wartosciami opcji
# "forecast" wektor z przewidywanymi wartosciami opcji
def option_price_pred(model, option_prices, stock_prices, rates, tag, time_start, pred_steps, lag, days_in_year):
    element = option_prices[(option_prices["tag"] == tag) & (option_prices["time_to_maturity"] <= time_start) & (option_prices["time_to_maturity"] > time_start - pred_steps)]
    start_date = element.head(1)["date"].values[0]

    if len(element) != pred_steps:
        return None

    option_type = element.head(1)["type"].values[0]
    strike = element.head(1)["strike"].values[0]
    train_prices = stock_prices[stock_prices["Data"] < start_date][-(lag):]["Zamkniecie"].values
    test_prices = element["price"].values
    sigma = gbm.train_gbm(train_prices)["sigma"]

    wimean_periods = [14, 30, 90, 180, 270, 365]
    rates = list(rates[rates["date"] < start_date].tail(1).values[0][1:])
    rate = np.interp(time_start, wimean_periods, rates)
    times = [x/float(days_in_year) for x in range(time_start, time_start - pred_steps, -1)]

    train_prices = np.reshape(train_prices, (-1, 1))
    stock_pred = stock_model.predict_recursive(model, train_prices, pred_steps)

    option_pred = np.array([black_scholes.euro_vanilla(float(stock_pred[i]), strike, times[i], rate, sigma, option_type) for i in range(pred_steps)])
    return {"actual": test_prices, "forecast": option_pred}

#przewidywania dla wielu "tagow"
#tags - tablica tagow
#reszta parametrow taka sama

#wartosc wyjsciowa - slownik z dwoma wpisami
# "actual": macierz o wymiarach (len(tags) x pred_steps) z prawdziwymi wartosciami opcji
# "forecast" macierz o wymiarach (len(tags) x pred_steps)z  przewidywanymi wartosciami opcji
def option_pred_matrix(model, option_prices, stock_prices, rates, tags, time_start, pred_steps, lag, days_in_year):
    actuals = []
    forecasts = []
    for tag in tags:
        fc = option_price_pred(model, option_prices, stock_prices, rates, tag, time_start, pred_steps, lag, days_in_year)
        if fc:
            actuals.append(fc["actual"])
            forecasts.append(fc["forecast"])
    return {"actuals": actuals, "forecasts": forecasts}