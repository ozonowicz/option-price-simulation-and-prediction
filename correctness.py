from numpy import mean

def create_correctness_matrix(actuals, forecasts, n_forecasts, spans, thresholds):
    result = {"spans": spans, "thresholds": thresholds, "correctness": {}, "mape": {}}
    for span in spans:
        for threshold in thresholds:
            dict_str = str(span) + "D_" + str(threshold)
            result["correctness"][dict_str] = 0

    for i in range(n_forecasts):
        for span in spans:
            actual = actuals[i]
            forecast = forecasts[i]
            err = [abs(actual[j]-forecast[j])/actual[j] for j in range(span)]
            mape = mean(err)
            if str(span) in result["mape"]:
                result["mape"][str(span)] = result["mape"][str(span)] + mape
            else:
                result["mape"][str(span)] = mape
            for threshold in thresholds:
                if all(e < threshold for e in err):
                    dict_str = str(span) + "D_" + str(threshold)
                    result["correctness"][dict_str] = result["correctness"][dict_str] + 1
    for element in result["correctness"]:
        result["correctness"][element] = result["correctness"][element]/n_forecasts
    for element in result["mape"]:
        result["mape"][element] = result["mape"][element]/n_forecasts
    return result

def print_correctness(corr):
    num_format = "%.2f"
    print("    |  MAPE  | ", end="")
    for thres in corr["thresholds"]:
        thres_str = "C[" + (num_format % thres) + "] "
        print(thres_str, end=" ")
    print("\n__________________________________________________\n")
    for span in corr["spans"]:
        print("{:-2d}D".format(span), end=" | ")
        print("{:-6.2%}".format(corr["mape"][str(span)]), end=" | ")
        for thres in corr["thresholds"]:
            dict_str = str(span) + "D_" + str(thres)
            print("{:-7.2%}".format(corr["correctness"][dict_str]), end="  ")
        print()
