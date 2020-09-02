import numpy
import math
import scipy.stats
import statistics
import random

random.seed(3191)
numpy.random.seed(3191)

def gbm_test(prices, method, p):
    log_returns = [math.log(prices[i+1]/prices[i]) for i in range(len(prices)-1)]
    if method == "shapiro":
        test = scipy.stats.shapiro(log_returns)
        return test[1] < p
    elif method == "kolmogorov":
        test = scipy.stats.kstest(log_returns, "norm", N=len(log_returns), mode="asymp")
        return test[1] < p

def gbm_random_path(mu, sigma, start, step, n):
    result = [start]
    for i in range(n):
        wiener = numpy.random.normal(0, 1)
        result.append(result[-1] * math.exp((mu - (sigma * sigma) / 2) * step + sigma * numpy.sqrt(step) * wiener))
    return result

def gbm_paths(mu, sigma, start, step, n, paths):
    return [gbm_random_path(mu, sigma, start, step, n) for i in range(paths)]

def median_path(paths):
    pth1 = [[numpy.linalg.norm(p), p] for p in paths]
    if len(pth1) % 2 == 0:
        pth1 = pth1[1:]
    return statistics.median(pth1)[1]

def closest_path_prediction(train_path, mu, sigma, step, n_paths, prediction_days):
    paths = gbm_paths(mu, sigma, train_path[0], step, len(train_path) + prediction_days, n_paths)
    path_differences = [ [ [path[i] - train_path[i]] for i in range(len(train_path))] for path in paths]

    path_distances = [[numpy.linalg.norm(path_differences[i]), paths[i][len(train_path):]] for i in range(len(paths))]
    return min(path_distances)[1]

def simulate_gbm(mu, sigma, start, step, n, paths, method = "median", train_path = None):
    if method == "median":
        return median_path(gbm_paths(mu, sigma, start, step, n, paths))
    else:  # method == closest_path
        return closest_path_prediction(train_path, mu, sigma, step, paths, n)


def train_gbm(prices, days_in_year=254, likelihood=False):
    if likelihood:
        daily_returns = [prices[i + 1] / prices[i] for i in range(len(prices) - 1)]
        log_returns = [math.log(a) for a in daily_returns]
        mu_ = 1.0*numpy.mean(log_returns)*days_in_year
        sigma_ = numpy.std(log_returns) * math.sqrt(days_in_year)
        return {"mu": mu_, "sigma": sigma_}
    else:
        daily_returns = [ (prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices) - 1)]
        mu_ = 1.0 * numpy.mean(daily_returns) * days_in_year
        sigma_ = numpy.std(daily_returns) * math.sqrt(days_in_year)
        return {"mu": mu_, "sigma": sigma_}