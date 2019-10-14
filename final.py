# Libraries
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA


class Gold:

    # Constructor
    def __init__(self):
        # Logarithm of gold prices
        self.__log_gold = [5.264967387, 5.719262046, 6.420808929, 6.129616498,
                           5.927725706, 6.048931247, 5.888268354, 5.759847699,
                           5.907675246, 6.100812104, 6.079612778, 5.942326823,
                           5.949496062, 5.892362186, 5.840496298, 5.885603906,
                           5.951033101, 5.950772752, 5.960670232, 5.802994125,
                           5.683885843, 5.629669374, 5.631570141, 5.602266411,
                           5.735539506, 5.895283989, 6.014130718, 6.096837563,
                           6.403193331, 6.544472839, 6.770743551, 6.879715822,
                           7.110304209, 7.359798583, 7.41996794, 7.252216944,
                           7.143933509]
        # Given gold prices
        self.__gold = [193, 305, 615, 459, 375, 424, 361, 317, 368, 446, 437,
                       381, 384, 362, 344, 360, 384, 384, 388, 331, 294, 279,
                       279, 271, 310, 363, 409, 444, 604, 695, 872, 972, 1225,
                       1572, 1669, 1411, 1266]

        self.__df = pd.DataFrame()
        self.__df["gold"] = self.__gold
        self.__df['log_gold'] = self.__log_gold
        self.__df["gold_lag"] = self.__df["gold"].shift()
        self.__df["log_gold_lag"] = self.__df["log_gold"].shift()
        self.__df.dropna(inplace=True)
        self.__df.reset_index(drop=True, inplace=True)
        self.__df["gold"] = self.__gold[1:]
        self.__df["log_gold"] = self.__log_gold[1:]

        self.__ppiaco = pd.read_csv('PPIACO.csv')
        self.__ppiaco.set_index('DATE', inplace=True)
        self.__ppiaco = self.__ppiaco['PPIACO']

        self.__will5000ind = pd.read_csv('WILL5000IND.csv')
        self.__will5000ind.set_index('DATE', inplace=True)
        self.__will5000ind = self.__will5000ind['WILL5000IND']

    def get_gold_data(self):
        return self.__df

    def implement_lm(self):
        regressor = sm.add_constant(self.__df["gold_lag"])
        model = sm.OLS(self.__df["gold"], regressor)
        fitted_model = model.fit()
        return fitted_model.summary()

    def implement_lm_log(self):
        regressor = sm.add_constant(self.__df["log_gold_lag"])
        model = sm.OLS(self.__df["log_gold"], regressor)
        fitted_model = model.fit()
        return fitted_model.summary()

    def implement_lm_wo_alpha(self):
        model = sm.OLS(self.__df["gold"], self.__df["gold_lag"])
        fitted_model = model.fit()
        return fitted_model.summary()

    def implement_2factor_lm(self):
        regressors = pd.DataFrame(np.log(self.__ppiaco))
        regressors['WILL5000IND'] = np.log(self.__will5000ind)
        regressors = sm.add_constant(regressors)
        model = sm.OLS(self.__log_gold, regressors)
        results = model.fit()
        predicted_values = np.exp(results.fittedvalues)
        print(results.summary())
        print 'forecast = ', np.exp(results.params[0] +
                                    results.params[1] * np.log(190.4) +
                                    results.params[2] * np.log(90.66))
        plt.title("Gold price during 1978-2014")
        plt.xlabel("Period")
        plt.ylabel("USD per troy ounce")
        plt.plot(predicted_values.values,  label="Predicted gold prices")
        plt.plot(self.__gold, label="Actual gold prices")
        plt.legend()
        plt.show()
        return predicted_values

    # Stationaruty test (Augmented Dickey-Fuller)
    def do_adf_test(self):
        print 'p_value in Augmented Dickey-Fuller test is',
        adfuller(self.__gold)[1]

    def implement_ARMA(self):
        # Plot ACF
        tsaplots.plot_acf(self.__gold)
        plt.show()

        # Plot PACF
        tsaplots.plot_pacf(self.__gold)
        plt.show()

        # ARIMA(2,0,0)
        gold = np.array(self.__gold, dtype=np.float)
        model = ARIMA(gold, order=(2, 0, 0))
        ARIMA200 = model.fit()
        print ARIMA200.summary()
        print 'Next price is :', ARIMA200.forecast()[0]
        predicted_values_ARIMA200 = ARIMA200.fittedvalues
        plt.title("Gold price during 1978-2014")
        plt.xlabel("Period")
        plt.ylabel("USD per troy ounce")
        plt.plot(predicted_values_ARIMA200,
                 label="Predicted gold prices ARIMA(2,0,0)")
        plt.plot(self.__gold, label="Actual gold prices")
        plt.legend()
        plt.show()
        return predicted_values_ARIMA200

    def main(self):
        gold.implement_lm()
        gold.implement_lm_log()
        gold.implement_lm_wo_alpha()
        gold.implement_2factor_lm()
        gold.do_adf_test()
        gold.implement_ARMA()


gold = Gold()
gold.main()
