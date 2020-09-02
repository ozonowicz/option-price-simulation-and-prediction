import numpy as np
from scipy.stats import norm
import gbm
import numpy

#oparte na:
#aaronschlegel.me/black-scholes-formula-python.html

def euro_vanilla(S, K, T, r, sigma, option = 'call'):
    
    #S: cena spot
    #K: cena wywolania
    #T: czas wygasniecie
    #r: stopa procentowa
    #sigma: dyfuzja/wrazliwosc instrumentu bazowego
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
        
    return result

def euro_vanilla_vectorized(S_vec, K, T_vec, r, sigma, option = 'call'):
    if len(T_vec) == 1:
        T_vec = T_vec * len(S_vec)
    n = min(len(S_vec), len(T_vec))
    return [euro_vanilla(S_vec[i], K, T_vec[i], r, sigma, option) for i in range(n)]

def train_bs(stock_prices, option_prices, times_to_maturity, strike, option_type):
    n = min(len(stock_prices), len(times_to_maturity))
    gbm_params = gbm.train_gbm(stock_prices, 254)
    sigma = gbm_params["sigma"]

    option_prices_for_different_interest_rates = []

    interest_rates = [x/10000 for x in range(1,2001)]

    for r in interest_rates:
        bs_prices = euro_vanilla_vectorized(stock_prices, strike, times_to_maturity, r, sigma, option_type)
        diff = [bs_prices[i] - option_prices[i] for i in range(n)]
        dist = numpy.linalg.norm(diff)
        option_prices_for_different_interest_rates.append([dist, r])

    option_prices_for_different_interest_rates.sort()
    r = option_prices_for_different_interest_rates.sort()[0][1]

    return {"sigma": sigma, "r": r}
