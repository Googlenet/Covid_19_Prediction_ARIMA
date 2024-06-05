import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima.arima import ADFTest
from sklearn.metrics import mean_absolute_error, \
    mean_absolute_percentage_error, mean_squared_error
import warnings

# ignore warnings that pop up
warnings.filterwarnings("ignore")

# preprocessing
raw_data = pd.read_csv('time_series_covid19_confirmed_US.csv')

# Shape of original dataframe
print('Shape of Original Data: {}'.format(raw_data.shape))
print()

# remove all columns that are not relevant to analysis
covid_data = raw_data.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS',
                            'Admin2', 'Country_Region', 'Lat', 'Long_',
                            'Combined_Key'], axis='columns')

# gets the set of all unique locations in the data
all_states = sorted(list(set(covid_data['Province_State'])))

# focus on one location at once due to time constraints, but code
# can be generalized to any other location
current_state = 'California'
covid_data = covid_data[covid_data['Province_State'] == current_state]

# group by state and combine values of all different areas together
state_data = covid_data.groupby(['Province_State']).sum()
state_data_row = state_data.iloc[0, :]

# set ticks for date to be displayed on plot
ticks = []
x_ranges = round(state_data.size / 10)
for i in range(0, state_data.size, x_ranges):
    ticks.append(state_data.columns[i])

# adjust padding to get last day in data
ticks[len(ticks) - 1] = state_data.columns[state_data.size - 1]

# plot data
fig = plt.figure()
plt.rcParams['figure.figsize'] = (12, 6)
plt.plot(state_data.columns, state_data_row, label='# Covid Cases')
plt.tick_params(axis='x', rotation=45)
plt.tick_params(axis='y', rotation=45)
plt.xticks(ticks)
plt.ticklabel_format(style='plain', axis='y')
plt.subplots_adjust(bottom=0.20)
plt.xlabel('Date (MM/DD/YYYY)')
plt.ylabel('# of Covid Cases')
plt.title('# of Covid Cases in {} Over Time'.format(current_state))
plt.legend()
plt.show()

# save plot
print('Saving plot \'# of Covid Cases in {} '
      'Over Time\' to project directory'.format(current_state))
print()
fig.savefig('# of Covid Cases in {} Over Time.jpg'.format(current_state))

# Use an Augmented Dickey-Fuller test to test for stationary
# p-value > 0.05: Fail to reject the null hypothesis (H0),
# the data has a unit root and is non-stationary.
# p-value <= 0.05: Reject the null hypothesis (H0),
# the data does not have a unit root and is stationary.
adf_test = ADFTest(alpha=0.05)
adf_p_val = adf_test.should_diff(state_data_row)
print('Augmented Dickey-Fuller test: {}'.format(adf_p_val))

# reject or accept null based on p-value
if adf_p_val[0] > 0.05:
    print('p-value > 0.05, fail to reject null hypothesis (H0), '
          'the data has a unit root and is non-stationary.')
elif adf_p_val[0] <= 0.05:
    print('p-value <= 0.05: Reject the null hypothesis (H0), '
          'the data does not have a unit root and is stationary.')
print()

# visually find the 'd' value for differencing by plotting original
# series, 1st and 2nd difference

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(state_data_row)
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)

# 1st Differencing
ax2.plot(state_data_row.diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)

# 2nd Differencing
ax3.plot(state_data_row.diff().diff())
ax3.set_title('2nd Order Differencing')
plt.xticks(ticks)
plt.tick_params(axis='x', rotation=45)
plt.show()

# save plot
print('Saving plot \'0-2 Orders of Differencing '
      'On Data\' to project directory')
print()
fig.savefig('0-2 Orders of Differencing On Data.jpg')

# now find 'p' by plotting pacf of 1st and 2nd order difference
fig, (ax1, ax2) = plt.subplots(2)

# PACF 1st Differencing
plot_pacf(state_data_row.diff().dropna(),
          ax=ax1, title='PACF 1st Order Differencing')

# PACF 2nd Differencing
plot_pacf(state_data_row.diff().diff().dropna(),
          ax=ax2, title='PACF 2nd Order Differencing')

# extra info for plot
fig.tight_layout(pad=5.0)
fig.supxlabel('# of Lags')
fig.supylabel('Correlation Coefficient')
plt.show()

# save plot
print('Saving plot \'PACF Plot with 1-2 Order Differencing\' to project directory')
print()
fig.savefig('PACF Plot with 1-2 Order Differencing.jpg')

# now find 'q' by plotting acf of 1st and 2nd order difference
fig, (ax1, ax2) = plt.subplots(2)

# 1st Differencing
plot_acf(state_data_row.diff().dropna(),
         ax=ax1, title='ACF 1st Order Differencing')

# 2nd Differencing
plot_acf(state_data_row.diff().diff().dropna(),
         ax=ax2, title='ACF 2nd Order Differencing')

# extra info for plot
fig.tight_layout(pad=5.0)
fig.supxlabel('# of Lags')
fig.supylabel('Correlation Coefficient')
plt.show()

# save plot
print('Saving plot \'ACF Plot with 1-2 Order '
      'Differencing\' to project directory')
print()
fig.savefig('ACF Plot with 1-2 Order Differencing.jpg')

# split train/test into 80/20
train_split = round(len(state_data_row) * 0.8)
test_split = round(len(state_data_row) * 0.2)

# output # of features being used for train/test
print('First {} features to be trained on'.format(train_split))
print('Last {} features to be tested on'.format(test_split))
print()
train = state_data_row[:train_split]
test = state_data_row[-test_split:]

# visually plot train and test data
fig = plt.figure()
plt.plot(train, label='train data')
plt.plot(test, label='test data')
plt.tick_params(axis='x', rotation=45)
plt.tick_params(axis='y', rotation=45)
plt.xticks(ticks)
plt.ticklabel_format(style='plain', axis='y')
plt.subplots_adjust(bottom=0.20)
plt.xlabel('Date (MM/DD/YYYY)')
plt.ylabel('# of Covid Cases')
plt.title('# of Covid Cases in {} Over Time '
          'Train-Test Split'.format(current_state))
plt.legend()
plt.show()

# save plot
print('Saving plot \'# of Covid Cases in {} Over Time '
      'Train-Test Split\' to project directory'.format(current_state))
print()
fig.savefig('# of Covid Cases in {} Over Time '
            'Train-Test Split.jpg'.format(current_state))

# set ticks for arima plots
arima_ticks = []
x_ranges = round(test_split / 10)
for i in range(0, test_split, x_ranges):
    arima_ticks.append(state_data[-test_split:].columns[i + train_split])

# adjust padding to get last day in data
arima_ticks.append(state_data.columns[state_data.size - 1])

# create model with manual value of p, d, q from visually looking at plots
# with one degree of differencing
model_1 = ARIMA(state_data_row, order=(9, 1, 21))
model_1_fit = model_1.fit()

# predict on test data with ARIMA(9, 1, 21)
model_pred_1 = pd.DataFrame(model_1_fit.predict(n_periods=test_split),
                            index=test.index)
model_pred_1.columns = ['predicted']

# output error metrics
print('ARIMA Model with parameters (9, 1, 21)')
print('--------------------------------------------------------')
print('Mean Absolute Error: {}'.format(
    round(mean_absolute_error(list(state_data_row[-test_split:]),
                              list(model_pred_1['predicted'])), 4)))

print('Mean Absolute Percentage Error: {}'.format(
    round(mean_absolute_percentage_error(list(state_data_row[-test_split:]),
                                         list(model_pred_1['predicted'])), 7)))

print('Root Mean Squared Error: {}'.format(
    round(mean_squared_error(list(state_data_row[-test_split:]),
                             list(model_pred_1['predicted']),
                             squared=False), 4)))
print()

# plot ARIMA(9, 1, 21) prediction and true value
fig = plt.figure(figsize=(8, 5))
plt.plot(test, label='test')
plt.plot(model_pred_1, label='predicted')
plt.tick_params(axis='x', rotation=45)
plt.tick_params(axis='y', rotation=45)
plt.xticks(arima_ticks)
plt.ticklabel_format(style='plain', axis='y')
plt.subplots_adjust(bottom=0.20)
plt.xlabel('Date (MM/DD/YYYY)')
plt.ylabel('# of Covid Cases')
plt.title('ARIMA(9, 1, 21) Predicted vs. Actual '
          'in {} Over Time'.format(current_state))
plt.legend()
plt.show()

# save plot
print('Saving plot \'ARIMA(9, 1, 21) Predicted vs. Actual '
      'in {} Over Time\' to project directory'.format(current_state))
print()
fig.savefig('ARIMA(9, 1, 21) Predicted vs. Actual '
            'in {} Over Time.jpg'.format(current_state))

# create model with manual value of p, d, q from visually looking at plots
# with two degrees of differencing
model_2 = ARIMA(state_data_row, order=(9, 2, 13))
model_2_fit = model_2.fit()

# predict on test data with ARIMA(9, 2, 13)
model_pred_2 = pd.DataFrame(model_2_fit.predict(n_periods=test_split),
                            index=test.index)
model_pred_2.columns = ['predicted']

# output error metrics
print('ARIMA Model with parameters (9, 2, 13)')
print('--------------------------------------------------------')
print('Mean Absolute Error: {}'.format(
    round(mean_absolute_error(list(state_data_row[-test_split:]),
                              list(model_pred_2['predicted'])), 4)))

print('Mean Absolute Percentage Error: {}'.format(
    round(mean_absolute_percentage_error(list(state_data_row[-test_split:]),
                                         list(model_pred_2['predicted'])), 7)))

print('Root Mean Squared Error: {}'.format(
    round(mean_squared_error(list(state_data_row[-test_split:]),
                             list(model_pred_2['predicted']),
                             squared=False), 4)))
print()

# plot ARIMA(9, 2, 13) prediction and true value
fig = plt.figure(figsize=(8, 5))
plt.plot(test, label='test')
plt.plot(model_pred_2, label='predicted')
plt.tick_params(axis='x', rotation=45)
plt.tick_params(axis='y', rotation=45)
plt.xticks(arima_ticks)
plt.ticklabel_format(style='plain', axis='y')
plt.subplots_adjust(bottom=0.20)
plt.xlabel('Date (MM/DD/YYYY)')
plt.ylabel('# of Covid Cases')
plt.title('ARIMA(9, 2, 13) Predicted vs. Actual '
          'in {} Over Time'.format(current_state))
plt.legend()
plt.show()

# save plot
print('Saving plot \'ARIMA(9, 2, 13) Predicted vs. Actual '
      'in {} Over Time\' to project directory'.format(current_state))
print()
fig.savefig('ARIMA(9, 2, 13) Predicted vs. Actual '
            'in {} Over Time.jpg'.format(current_state))
