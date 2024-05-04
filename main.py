#!/usr/bin/env python3

import datatools as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
from sklearn import svm

if __name__ == "__main__":
    path = "Statewide_Greenhouse_Gas_Emissions__Beginning_1990.csv"
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", type=str)
        path = vars(parser.parse_args())["file"]
    data = dt.EmissionsData(path)

    yearly_emissions_mt20 = data.get_yearly_emissions()
    plt.hist(yearly_emissions_mt20, bins=dt.binsize(yearly_emissions_mt20, method="Rice's"), edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("MT CO2e AR5 20 yr")
    plt.title("Distribution of yearly MT CO2e AR5 20 yr emissions (Total)")
    mean = np.mean(yearly_emissions_mt20)
    median = np.median(yearly_emissions_mt20)
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print("Std: {0}\n".format(np.std(yearly_emissions_mt20)))
    q3 = np.percentile(yearly_emissions_mt20, 75)
    q1 = np.percentile(yearly_emissions_mt20, 25)
    plt.plot([q1, q3], [-0.1, -0.1], color="orange")
    plt.scatter(mean, -0.2, color="red", marker="^",label="Mean")
    plt.scatter(median, -0.3, color="green", marker="^", label="Median")
    plt.legend()
    plt.show()

    yearly_emissions_mt100 = data.get_yearly_emissions(metric="MT CO2e AR4 100 yr")
    plt.hist(yearly_emissions_mt100, bins=dt.binsize(yearly_emissions_mt100, method="Rice's"), edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("MT CO2e AR4 100 yr")
    plt.title("Distribution of yearly MT CO2e AR4 100 yr emissions (Total)")
    mean = np.mean(yearly_emissions_mt100)
    median = np.median(yearly_emissions_mt100)
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print("Std: {0}\n".format(np.std(yearly_emissions_mt100)))
    q3 = np.percentile(yearly_emissions_mt100, 75)
    q1 = np.percentile(yearly_emissions_mt100, 25)
    plt.plot([q1, q3], [-0.1, -0.1], color="orange")
    plt.scatter(mean, -0.2, color="red", marker="^",label="Mean")
    plt.scatter(median, -0.3, color="green", marker="^", label="Median")
    plt.legend()
    plt.show()

    yearly_removals_mt20 = data.get_yearly_removals()
    plt.hist(yearly_removals_mt20, bins=dt.binsize(yearly_removals_mt20, method="Rice's"), edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("MT CO2e AR5 20 yr removals")
    plt.title("Distribution of yearly MT CO2e AR5 20 yr emission removals (Total)")
    mean = np.mean(yearly_removals_mt20)
    median = np.median(yearly_removals_mt20)
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print("Std: {0}\n".format(np.std(yearly_removals_mt20)))
    q3 = np.percentile(yearly_removals_mt20, 75)
    q1 = np.percentile(yearly_removals_mt20, 25)
    plt.plot([q1, q3], [-0.1, -0.1], color="orange")
    plt.scatter(mean, -0.2, color="red", marker="^",label="Mean")
    plt.scatter(median, -0.3, color="green", marker="^", label="Median")
    plt.legend()
    plt.show()

    yearly_removals_mt100 = data.get_yearly_removals(metric="MT CO2e AR4 100 yr")
    plt.hist(yearly_removals_mt100, bins=dt.binsize(yearly_removals_mt100, method="Rice's"), edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("MT CO2e AR4 100 yr removals")
    plt.title("Distribution of yearly MT CO2e AR4 100 yr emission removals (Total)")
    mean = np.mean(yearly_removals_mt100)
    median = np.median(yearly_removals_mt100)
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print("Std: {0}\n".format(np.std(yearly_removals_mt100)))
    q3 = np.percentile(yearly_removals_mt100, 75)
    q1 = np.percentile(yearly_removals_mt100, 25)
    plt.plot([q1, q3], [-0.1, -0.1], color="orange")
    plt.scatter(mean, -0.2, color="red", marker="^",label="Mean")
    plt.scatter(median, -0.3, color="green", marker="^", label="Median")
    plt.legend()
    plt.show()

    yearly_emissions_mt20_by_gas = {}
    for gas in data.gasses:
        yearly_emissions_mt20_by_gas[gas] = data.get_yearly_emissions(gas=gas)
    stacked_bar = pd.DataFrame(yearly_emissions_mt20_by_gas, index=data.years)
    stacked_bar.plot(kind="bar", stacked=True)
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR5 20 yr")
    plt.title("MT CO2e AR5 20 yr emissions by gas")
    plt.legend(loc="lower center")
    plt.show()

    yearly_emissions_mt100_by_gas = {}
    for gas in data.gasses:
        yearly_emissions_mt100_by_gas[gas] = data.get_yearly_emissions(metric="MT CO2e AR4 100 yr", gas=gas)
    stacked_bar = pd.DataFrame(yearly_emissions_mt100_by_gas, index=data.years)
    stacked_bar.plot(kind="bar", stacked=True)
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR4 100 yr")
    plt.title("MT CO2e AR4 100 yr emissions by gas")
    plt.legend(loc="lower center")
    plt.show()

    yearly_removals_mt20_by_gas = {}
    for gas in data.gasses:
        yearly_removals_mt20_by_gas[gas] = abs(data.get_yearly_removals(gas=gas))
    stacked_bar = pd.DataFrame(yearly_removals_mt20_by_gas, index=data.years)
    stacked_bar.plot(kind="bar", stacked=True)
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR5 20 yr (removed)")
    plt.title("MT CO2e AR5 20 yr removals by gas")
    plt.legend(loc="lower center")
    plt.show()

    yearly_removals_mt100_by_gas = {}
    for gas in data.gasses:
        yearly_removals_mt100_by_gas[gas] = abs(data.get_yearly_removals(metric="MT CO2e AR4 100 yr", gas=gas))
    stacked_bar = pd.DataFrame(yearly_removals_mt100_by_gas, index=data.years)
    stacked_bar.plot(kind="bar", stacked=True)
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR4 100 yr (removed)")
    plt.title("MT CO2e AR4 100 yr removals by gas")
    plt.legend(loc="lower center")
    plt.show()

    plt.title("MT CO2e AR5 20 yr emissions by Economic Sector")
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR5 20 yr")
    for economic_sector in data.economic_sectors:
        emissions = data.get_yearly_emissions(economic_sector=economic_sector)
        if all(emissions == 0):
            continue
        plt.plot(data.years, emissions, label=economic_sector)
    plt.legend()
    plt.show()

    plt.title("MT CO2e AR5 20 yr emissions by Sector")
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR5 20 yr")
    for sector in data.sectors:
        emissions = data.get_yearly_emissions(sector=sector)
        if all(emissions == 0):
            continue
        plt.plot(data.years, emissions, label=sector)
    plt.legend()
    plt.show()

    plt.title("MT CO2e AR5 20 yr emissions by Category")
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR5 20 yr")
    for category in data.categories:
        emissions = data.get_yearly_emissions(category=category)
        if all(emissions == 0):
            continue
        plt.plot(data.years, emissions, label=category)
    plt.legend(prop={'size': 6})
    plt.show()

    plt.title("MT CO2e AR5 20 yr removals by Sub-Category 3")
    plt.xlabel("Years")
    plt.ylabel("MT CO2e AR5 20 yr (removed)")
    for subcategory3 in data.third_subcategories:
        removals = abs(data.get_yearly_removals(subcategory3=subcategory3))
        if all(removals == 0):
            continue
        plt.plot(data.years, removals, label=subcategory3)
    plt.legend(prop={'size': 7})
    plt.show()

    # Call the function below to programatically generate all possible graphs
    # The total_type takes:
    # - "total" (default) - To indicate you want to see the net emissions/removals (removals are along -y)
    # - "gross" - To indicate you want to see only the emissions and no removals
    # - "removals" - To indicate you only want to see the emission removals (removals are along +y)
    # Pass a iterable of years to see only a specified range of emission values with the year= kwarg (default is the unique years
    # in the dataset)
    # You can also select a specific metric by metric= kwarg. Default is MT CO2e AR5 20 yr
    # Likewise you can filter for a specific gas and indicate you want to only record datapoints consistent with the UN National
    # Framework Convention on Climate Change, using the gas=(str) and conventional=(bool) kwargs respectively
    # CAUTION, IS NOT EFFICIENT
    #dt.generate_granular_graphs(data)


    # Checking the Climate Leadership and Community Protection Act's goal of 60% of 1990 gross emissions 
    # by 2030 and 15% of 1990 gross emissions by 2050 (Using MT CO2e AR5 20 yr)
    emissions = data.get_yearly_gross_emissions()
    emissions_1990 = emissions[0]
    print(f"MT CO2e AR5 20 yr gross in 1990: {emissions_1990}\n")

    # Simple linear regression
    regression = sklearn.linear_model.LinearRegression()
    regression.fit(data.years.reshape((-1, 1)), emissions)
    plt.plot(range(data.years[0], data.years[-1] + 1), regression.predict(np.array(range(data.years[0], data.years[-1] + 1)).reshape((-1, 1))), color="red")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    plt.plot(range(1990, 2051), regression.predict(np.array(range(1990, 2051)).reshape((-1, 1))), color="red")
    simple_linear_prediction_2030 = regression.predict([[2030]])[0]
    plt.scatter(2030, simple_linear_prediction_2030, color="yellow")
    simple_linear_prediction_2050 = regression.predict([[2050]])[0]
    plt.scatter(2050, simple_linear_prediction_2050, color="yellow")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    print(f"Simple linear prediction of 2030 gross: {simple_linear_prediction_2030: .2}")
    print(f"Percentage of 1990: {(simple_linear_prediction_2030 / emissions_1990): .2%}")
    print(f"Simple linear prediction of 2030 gross: {simple_linear_prediction_2050: .2}")
    print(f"Percentage of 1990: {(simple_linear_prediction_2050 / emissions_1990): .2%}")
    # Standardized residuals
    predictions = regression.predict(data.years.reshape((-1, 1)))
    residuals = emissions - predictions
    std_residuals = residuals / np.std(residuals)
    plt.scatter(predictions, std_residuals, color="red")
    plt.axhline(0, linestyle="--", color="black")
    plt.ylabel("Standardized residuals")
    plt.xlabel("MT CO2e AR5 20 yr")
    plt.show()
    linear_rmse = np.std(residuals)
    print(f"Simple linear RMSE: {linear_rmse: .2}\n")

    # Polynomial regression
    # Brute force procedure used to find a good polynomial order
    # order_heuristics = np.zeros(5)
    # for order in range(1, 6):
    #     polynomial = sklearn.preprocessing.PolynomialFeatures(degree=order, include_bias=False)
    #     polynomials = polynomial.fit_transform(data.years.reshape((-1, 1)))
    #     polynomial_regression = sklearn.linear_model.LinearRegression()
    #     polynomial_regression.fit(polynomials, emissions)
    #     # Residual sum of squares
    #     rss = np.sum((emissions - polynomial_regression.predict(polynomials)) ** 2)
    #     # Our heuristic
    #     heuristic = rss / (len(data.years) - order - 1)
    #     order_heuristics[order - 1] = heuristic
    # print(order_heuristics)
    # From the heuristics, a polynomial order of 3 seems like a good predictor so we will try that
    polynomial = sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)
    polynomials = polynomial.fit_transform(data.years.reshape((-1, 1)))
    polynomial_regression = sklearn.linear_model.LinearRegression()
    polynomial_regression.fit(polynomials, emissions)
    third_order_predictions = polynomial_regression.predict(polynomials)
    plt.plot(range(data.years[0], data.years[-1] + 1), third_order_predictions, color="red")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    polynomials = polynomial.fit_transform(np.array(range(1990, 2051)).reshape((-1, 1)))
    plt.plot(range(1990, 2051), polynomial_regression.predict(polynomials), color="red")
    polynomials = polynomial.fit_transform([[2030]])
    polynomial_prediction_2030 = polynomial_regression.predict(polynomials)[0]
    plt.scatter(2030, polynomial_prediction_2030, color="yellow")
    polynomials = polynomial.fit_transform([[2050]])
    polynomial_prediction_2050 = polynomial_regression.predict(polynomials)[0]
    plt.scatter(2050, polynomial_prediction_2050, color="yellow")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    print(f"3rd order polynomial prediction of 2030 gross: {polynomial_prediction_2030: .2}")
    print(f"Percentage of 1990: {(polynomial_prediction_2030 / emissions_1990): .2%}")
    print(f"3rd order polynomial prediction of 2050 gross: {polynomial_prediction_2050: .2}")
    print(f"Percentage of 1990: {(polynomial_prediction_2050 / emissions_1990): .2%}")
    # Standardized residuals
    polynomials = polynomial.fit_transform(data.years.reshape((-1, 1)))
    predictions = polynomial_regression.predict(polynomials)
    residuals = emissions - predictions
    std_residuals = residuals / np.std(residuals)
    plt.scatter(predictions, std_residuals, color="red")
    plt.axhline(0, linestyle="--", color="black")
    plt.ylabel("Standardized residuals")
    plt.xlabel("MT CO2e AR5 20 yr")
    plt.show()
    plt.hist(residuals, edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("Standardized residual")
    plt.show()
    polynomial_model_3_rmse = np.std(residuals)
    print(f"3rd order polynomial RMSE: {polynomial_model_3_rmse: .2}\n")
    
    
    # We would have gotten much different extrapolation with degree 2!
    polynomial = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    polynomials = polynomial.fit_transform(data.years.reshape((-1, 1)))
    polynomial_regression = sklearn.linear_model.LinearRegression()
    polynomial_regression.fit(polynomials, emissions)
    second_order_predictions = polynomial_regression.predict(polynomials)
    plt.plot(range(data.years[0], data.years[-1] + 1), second_order_predictions, color="red")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    polynomials = polynomial.fit_transform(np.array(range(1990, 2051)).reshape((-1, 1)))
    plt.plot(range(1990, 2051), polynomial_regression.predict(polynomials), color="red")
    polynomials = polynomial.fit_transform([[2030]])
    polynomial_prediction_2030 = polynomial_regression.predict(polynomials)[0]
    plt.scatter(2030, polynomial_prediction_2030, color="yellow")
    polynomials = polynomial.fit_transform([[2050]])
    polynomial_prediction_2050 = polynomial_regression.predict(polynomials)[0]
    plt.scatter(2050, polynomial_prediction_2050, color="yellow")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    print(f"2nd order polynomial prediction of 2030 gross: {polynomial_prediction_2030: .2}")
    print(f"Percentage of 1990: {(polynomial_prediction_2030 / emissions_1990): .2%}")
    print(f"2nd order polynomial prediction of 2050 gross: {polynomial_prediction_2050: .2}")
    print(f"Percentage of 1990: {(polynomial_prediction_2050 / emissions_1990): .2%}")
    # Standardized residuals
    polynomials = polynomial.fit_transform(data.years.reshape((-1, 1)))
    predictions = polynomial_regression.predict(polynomials)
    residuals = emissions - predictions
    std_residuals = residuals / np.std(residuals)
    plt.scatter(predictions, std_residuals, color="red")
    plt.axhline(0, linestyle="--", color="black")
    plt.ylabel("Standardized residuals")
    plt.xlabel("MT CO2e AR5 20 yr")
    plt.show()
    plt.hist(residuals, edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("Standardized residual")
    plt.show()
    polynomial_model_2_rmse = np.std(residuals)
    print(f"2nd order polynomial RMSE: {polynomial_model_2_rmse: .2}\n")

    svr =  svm.SVR(C=1e9)
    svr.fit(np.array(range(data.years[0], data.years[-1] + 1)).reshape((-1, 1)), emissions)
    svr_predictions = svr.predict(np.array(range(data.years[0], data.years[-1] + 1)).reshape((-1, 1)))
    plt.plot(range(data.years[0], data.years[-1] + 1), svr_predictions, color="red")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    plt.plot(range(1990, 2051), svr.predict(np.array(range(1990, 2051)).reshape((-1, 1))), color="red")
    svr_prediction_2030 = svr.predict([[2030]])[0]
    plt.scatter(2030, svr_prediction_2030, color="yellow")
    svr_prediction_2050 = svr.predict([[2050]])[0]
    plt.scatter(2050, svr_prediction_2050, color="yellow")
    plt.scatter(data.years, emissions)
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()   
    print(f"svr prediction of 2030 gross: {svr_prediction_2030: .2}")
    print(f"Percentage of 1990: {(svr_prediction_2030 / emissions_1990): .2%}")
    print(f"svr prediction of 2050 gross: {svr_prediction_2050: .2}")
    print(f"Percentage of 1990: {(svr_prediction_2050 / emissions_1990): .2%}")
    residuals = emissions - svr_predictions 
    std_residuals = residuals / np.std(residuals)
    plt.scatter(svr_predictions, std_residuals, color="red")
    plt.axhline(0, linestyle="--", color="black")
    plt.ylabel("Standardized residuals")
    plt.xlabel("MT CO2e AR5 20 yr")
    plt.show()
    plt.hist(residuals, edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("Standardized residual")
    plt.show()
    svr_model_2_rmse = np.std(residuals)
    print(f"svr RMSE: {svr_model_2_rmse: .2}\n")

    # Testing models extrapolation/variance with 70/15/15 split
    last_years = np.flip(data.years[:len(data.years) - 1 - int(len(data.years)*.15):-1])
    last_emissions = np.flip(emissions[:len(emissions) - 1 - int(len(data.years)*.15):-1])
    avail_years = data.years[:len(data.years) - int(len(data.years)*.15)]
    avail_emissions = emissions[:len(emissions) - int(len(emissions)*.15)]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(avail_years.reshape((-1,1)), avail_emissions, test_size=.15)

    regression.fit(X_train, y_train)
    plt.plot(avail_years, regression.predict(avail_years.reshape((-1, 1))), color="red")
    plt.scatter(avail_years, avail_emissions)
    plt.scatter(last_years, last_emissions, color="green")
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    new_linear_predictions = regression.predict(avail_years.reshape((-1, 1)))
    residuals = avail_emissions - new_linear_predictions
    new_linear_rmse = np.std(residuals)
    new_linear_extrapolation_predictions = regression.predict(last_years.reshape((-1, 1)))
    residuals = last_emissions - new_linear_extrapolation_predictions
    new_linear_extrapolation_rmse = np.std(residuals)
    new_linear_extrapolation_change = new_linear_extrapolation_rmse / new_linear_rmse
    print(f"New linear model RMSE: {new_linear_rmse: .2}")
    print(f"New linear model extrapolation RMSE: {new_linear_extrapolation_rmse: .2}")
    print(f"Change: {new_linear_extrapolation_change: .2}x\n")

    polynomial = sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)
    polynomials = polynomial.fit_transform(X_train)
    polynomial_regression = sklearn.linear_model.LinearRegression()
    polynomial_regression.fit(polynomials, y_train)
    polynomials = polynomial.fit_transform(avail_years.reshape((-1,1)))
    plt.plot(avail_years, polynomial_regression.predict(polynomials), color="red")
    plt.scatter(avail_years, avail_emissions)
    plt.scatter(last_years, last_emissions, color="green")
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    new_third_order_predictions = polynomial_regression.predict(polynomials)
    residuals = avail_emissions - new_third_order_predictions
    new_third_order_rmse = np.std(residuals)
    new_third_order_extrapolation_predictions = regression.predict(last_years.reshape((-1, 1)))
    residuals = last_emissions - new_third_order_extrapolation_predictions
    new_third_order_extrapolation_rmse = np.std(residuals)
    new_third_order_extrapolation_change = new_third_order_extrapolation_rmse / new_third_order_rmse
    print(f"New third order model RMSE: {new_third_order_rmse: .2}")
    print(f"New third order model extrapolation RMSE: {new_third_order_extrapolation_rmse: .2}")
    print(f"Change: {new_third_order_extrapolation_change: .2}x\n")

    polynomial = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    polynomials = polynomial.fit_transform(X_train)
    polynomial_regression = sklearn.linear_model.LinearRegression()
    polynomial_regression.fit(polynomials, y_train)
    polynomials = polynomial.fit_transform(avail_years.reshape((-1,1)))
    plt.plot(avail_years, polynomial_regression.predict(polynomials), color="red")
    plt.scatter(avail_years, avail_emissions)
    plt.scatter(last_years, last_emissions, color="green")
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    new_second_order_predictions = polynomial_regression.predict(polynomials)
    residuals = avail_emissions - new_second_order_predictions
    new_second_order_rmse = np.std(residuals)
    new_second_order_extrapolation_predictions = regression.predict(last_years.reshape((-1, 1)))
    residuals = last_emissions - new_second_order_extrapolation_predictions
    new_second_order_extrapolation_rmse = np.std(residuals)
    new_second_order_extrapolation_change = new_second_order_extrapolation_rmse / new_second_order_rmse
    print(f"New second order model RMSE: {new_second_order_rmse: .2}")
    print(f"New third order model extrapolation RMSE: {new_second_order_extrapolation_rmse: .2}")
    print(f"Change: {new_second_order_extrapolation_change: .2}x\n")

    svr.fit(X_train, y_train)
    plt.plot(avail_years, svr.predict(avail_years.reshape((-1, 1))), color="red")
    plt.scatter(avail_years, avail_emissions)
    plt.scatter(last_years, last_emissions, color="green")
    plt.xlabel("Years")
    plt.ylabel("Gross MT CO2e AR5 20 yr")
    plt.show()
    new_svr_predictions = svr.predict(avail_years.reshape((-1, 1)))
    residuals = avail_emissions - new_svr_predictions
    new_svr_rmse = np.std(residuals)
    new_svr_extrapolation_predictions = svr.predict(last_years.reshape((-1, 1)))
    residuals = last_emissions - new_svr_extrapolation_predictions
    new_svr_extrapolation_rmse = np.std(residuals)
    new_svr_extrapolation_change = new_svr_extrapolation_rmse / new_svr_rmse
    print(f"New svr model RMSE: {new_svr_rmse: .2}")
    print(f"New svr model extrapolation RMSE: {new_svr_extrapolation_rmse: .2}")
    print(f"Change: {new_svr_extrapolation_change: .2}x\n")

    print(np.argmax([new_linear_rmse, new_second_order_rmse, new_third_order_rmse, new_svr_rmse]))