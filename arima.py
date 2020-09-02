from statsmodels.tsa.arima_model import ARIMA
import random
import numpy as np

random.seed(319)
np.random.seed(319)


def arima_model_fit(train, order_):
    model = ARIMA(train, order = order_)
    return model.fit(disp=0)

def arima_forecast(train, steps, order_):
    model = ARIMA(train, order=order_)
    model_fit = model.fit(disp=0)
    return list(model_fit.forecast(steps)[0])