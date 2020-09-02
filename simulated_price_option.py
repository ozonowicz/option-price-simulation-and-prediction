from simulation_utils import *
import black_scholes
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def make_plot(par, title_tail, filename, scatter = False):
    title = "wartości błędu przy różnych " + title_tail
    fig, ax = plt.subplots()
    if scatter:
        ax.scatter(par["vals"], par["mean"], marker='.', color='red', label="średnia błędu")
        ax.scatter(par["vals"], par["sd"], marker='.', color='blue', label="odch. stand. błędu")
    else:
        ax.plot(par["vals"], par["mean"], marker='', color='red', linewidth=2, label="średnia błędu")
        ax.plot(par["vals"], par["sd"], marker='', color='blue', linewidth=2, label="odch. stand. błędu")
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Wartość parametru")
    ax.set_ylabel("Błąd względny [%]")
    ax.set_title(title)
    fig.savefig(filename)

#pojedyncza symulacja

#parametry
mu = 0.04
sigma = 0.25
stock_price = 1000
strike = 1020
option_type = "call"
r = 0.05
days = 30

#cena symulowana
print(option_price_simulated(mu, sigma, stock_price, strike, option_type, r, days))
#cena black scholes
print(black_scholes.euro_vanilla(stock_price, strike, days/254, r, sigma, option_type))


###########
#symulacja dla różnych wartości parametrów - wykresy
#tutaj - wzięto wartości mu od -0.1, 0,09, ..., 0, 0.1, ..., 0.9, 1
#dla każdej wylosowano 100 wektorów pozostałych parametrów, oraz obliczono średnią i odchylenie standardowe
par_mu = parameter_error_function(100, "mu", [x/100 for x in range(-10, 11)])
make_plot(par_mu, "wartościach parametru    mu", "sim_mu.png")

par_sigma = parameter_error_function(100, "sigma", [x/100 for x in range(2, 31, 2)])
make_plot(par_sigma, "wartościach parametru sigma", "sim_sigma.png")

par_strike = parameter_error_function(100, "relative_strike", [x/100 for x in range(-10, 11)])
make_plot(par_strike, "stosunkach strike'u do ceny akcji", "sim_strike.png")

par_type = parameter_error_function(100, "option_type", ["call", "put"])
make_plot(par_type, "typach opcji", "sim_type.png", scatter=True)

par_r = parameter_error_function(100, "r", [x/100 for x in range(11)])
make_plot(par_r, "stopach procentowych", "sim_rate.png")

par_days = parameter_error_function(100, "days", range(2, 101, 2))
make_plot(par_days, "terminach do wygaśnięcia", "sim_days.png")
