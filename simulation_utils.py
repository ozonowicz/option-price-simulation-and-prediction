import numpy as np
import gbm
import random
import black_scholes
import arima

def calculate_mape(actual, forecast):
    n = min(len(actual), len(forecast))
    return np.mean([abs((actual[i]-forecast[i])/actual[i]) for i in range(n)])


def generate_stock_prices(mu, sigma, learn_days, test_days):
    result = {}
    gbm_path = gbm.gbm_random_path(mu, sigma, 1000, 1/254, learn_days + test_days)
    result["learn"] = gbm_path[:learn_days]
    result["test"] = gbm_path[learn_days:]

def single_payoff(price, strike, type):
    if type == "call":
        return max(price-strike, 0)
    if type == "put":
        return max(strike-price, 0)

def path_payoff(path, strike, option_type):
    return [single_payoff(x, strike, option_type) for x in path]

def multiple_paths_payoffs(paths, strike, option_type):
    return [path_payoff(path, strike, option_type) for path in paths]

def averaged_payoff(paths, strike, type, r, n):
    avg_payoff = np.mean([single_payoff(path[-1], strike, type) for path in paths])
    discount_factor = np.exp(-r * n / 365)
    return avg_payoff * discount_factor

def option_price_simulated(mu, sigma, stock_price, strike, type, r, days_to_maturity):
    paths = gbm.gbm_paths(mu, sigma, stock_price, 1/254, days_to_maturity, 100)
    return averaged_payoff(paths, strike, type, r, days_to_maturity)

def generate_parameter_matrix(n, arg, val):
    result = []
    for i in range(n):
        row = dict(mu=random.uniform(-0.1, 0.1),
                   sigma=random.uniform(0.05, 0.3),
                   relative_strike=random.uniform(-0.1, 0.1),
                   option_type=random.choice(["call", "put"]),
                   r=random.uniform(0, 0.1),
                   days=random.choice(range(1, 101)))
        row[arg] = val
        result.append(row)
    return result

def parameters_error(params):
    bs_price = black_scholes.euro_vanilla(1000,
                                          1000*(1 + params["relative_strike"]),
                                          params["days"]/254,
                                          params["r"],
                                          params["sigma"],
                                          params["option_type"])

    sim_price = option_price_simulated(params["mu"],
                                       params["sigma"],
                                       1000,
                                       1000*(1 + params["relative_strike"]),
                                       params["option_type"],
                                       params["r"],
                                       params["days"])

    bs_price = min(0.01, bs_price)
    sim_price = min(0.01, sim_price)

    return abs(bs_price - sim_price)/bs_price

def parameter_errors(n, arg, val):
    try:
        mtx = generate_parameter_matrix(n, arg, val)
        errors = [parameters_error(x) for x in mtx]
    except:
        print(str(n) + " " + str(arg) + " " + str(val))

    return {"mean": np.mean(errors), "sd": np.std(errors)}

def parameter_error_function(n_rand, arg, vals):
    result = {"vals": vals, "mean": [], "sd": []}
    for val in vals:
        err = parameter_errors(n_rand, arg, val)
        result["mean"].append(err["mean"])
        result["sd"].append(err["sd"])
    return result

def generate_option_price_path(n_days, stock_start, mu, sigma, type, strike, r):
    fundamental = gbm.gbm_random_path(mu, sigma, stock_start, 1 / 254, n_days)
    path = []
    for i in range(n_days):
        days_to_maturity = n_days - i
        price = option_price_simulated(mu, sigma, fundamental[i+1], strike, type, r, days_to_maturity)
        path.append(max(price, 0.01))
        #path.append(black_scholes.euro_vanilla(fundamental[i+1], strike, days_to_maturity/254, r, sigma, type))
    return path

def train_and_test_generate(n_train, n_test, stock_start, mu, sigma, type, strike, r):
    train = gbm.gbm_random_path(mu, sigma, stock_start, 1 / 254, n_train)
    test = generate_option_price_path(n_test, train[-1], mu, sigma, type, strike, r)
    return {"train": train, "test": test}

def simulated_pred(train_prices, test_prices, strike, r, type):
    parameters = gbm.train_gbm(train_prices)
    random.seed(14100)
    np.random.seed(14100)
    mu = parameters["mu"]
    sigma = parameters["sigma"]
    n_pred = len(test_prices)
    #stock_pred = gbm.simulate_gbm(mu, sigma, train_prices[-1], 1/254, len(test_prices), 1000)
    stock_pred = arima.arima_forecast(train_prices, n_pred, (0,1,2))
    time_process = [x/254  for x in range(n_pred, 0, -1)]
    return black_scholes.euro_vanilla_vectorized(stock_pred, strike, time_process, r, sigma, type)
