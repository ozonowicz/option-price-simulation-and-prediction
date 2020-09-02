from simulation_utils import *
from correctness import *

forecasts = []
actuals = []
n_forecasts = 100

for i in range(n_forecasts):
    random.seed(i)
    np.random.seed(i)
    prices = train_and_test_generate(254, 25, 1000, 0.1, 0.3, 'call', 1060, 0.03)

    forecast = simulated_pred(prices["train"], prices["test"], 1060, 0.03, "call")
    actuals.append(prices["test"])
    forecasts.append(forecast)
    #print(i)

corr = create_correctness_matrix(actuals, forecasts, n_forecasts, [1,2,5,10], [0.05, 0.1, 0.2, 0.5])
print_correctness(corr)