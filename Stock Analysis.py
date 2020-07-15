#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np


# In[86]:


df = pd.read_csv("GOOG.csv")
df.head()


# In[87]:


df.columns


# In[88]:


df.describe()


# In[89]:


df.shape
df['Total Traded Quantity'].value_counts()


# In[90]:


df['Close Price'] = df['Close Price'].str.replace(',','').astype('float64')


# In[91]:


df['Total Traded Quantity'].head()


# In[92]:


df['Total Traded Quantity'] = df['Total Traded Quantity'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int)


# In[93]:


i = df['Close Price'].iloc[:90]
max_price = i.max()
min_price = i.min()
mean_price = i.mean()
print(max_price, min_price, mean_price)


# In[94]:


df['Date'] = df['Date'].astype('datetime64[ns]')
df['Date'].max() - df['Date'].min()


# In[95]:


df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Year'] = pd.DatetimeIndex(df['Date']).year
df.head()


# VWAP stands for Volume Weighted Average Price. It is a measure of the average price at which a stock is trading over the trading horizon. It is calculated by multiplying the price of the stock with the total volume or the total traded quantity of the stock and then dividing the whole by the total traded quantity. 

# In[96]:


def vwap(df):
    q = df['Close Price']
    p = df['Total Traded Quantity']
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())

df = df.groupby(['Month','Year'], group_keys=False).apply(vwap)
df.head()


# The below function is to calculate the average closing price of the stock for N number of days. The average price has been calculated for 1 week, 2 weeks, 1 month, 3 months, 6 months and 1 year respectively.

# In[97]:


def avg(N):
    avg_price = df['Close Price'].iloc[:N].mean()
    return avg_price


# In[98]:


avg_1 = avg(7)
avg_2 = avg(14)
avg_3 = avg(30)
avg_4 = avg(90)
avg_5 = avg(180)
avg_6 = avg(365)


# In[99]:


print('Average Price in : ')
print('1 week :', format(avg_1))
print('2 week :', format(avg_2))
print('1 month :', format(avg_3))
print('3 month :', format(avg_4))
print('6 month :', format(avg_5))
print('1 year :', format(avg_6))


# The below function is to calculate the profit or loss for the N number of days. The profit/ loss has been calculated for 1 week, 2 weeks, 1 month, 3 months, 6 months and 1 year respectively.

# In[100]:


def pl(N):
    per = ((df['Close Price'].iloc[N] - df['Close Price'].iloc[0])/df['Close Price'].iloc[N])*100
    return per


# In[101]:


per_1 = pl(7)
per_2 = pl(14)
per_3 = pl(30)
per_4 = pl(90)
per_5 = pl(120)
per_6 = pl(365)


# In[102]:


print('Profit or Loss in : ')
print('1 week :', format(per_1))
print('2 week :', format(per_2))
print('1 month :', format(per_3))
print('3 month :', format(per_4))
print('6 month :', format(per_5))
print('1 year :', format(per_6))


# Next the trends of the stock is observed based on the daily returns i.e. daily percentage change in closing price of the stock. The trends are analyzed based on the following conditions :-
# <br> 1. If the daily returns are between -0.5 and 0.5 that means very slight change or no change
# <br> 2. If the daily returns are between 0.5 and 1 that means slight change on the positive side
# <br> 3. If the daily returns are between -1 and -0.5 that means slight change on the negative side
# <br> 4. If the daily returns are between 1 and 3 that means change on the positive side
# <br> 5. If the daily returns are between -3 and -1 that means change on the negative side
# <br> 6. If the daily returns are between 3 and 7 that means top gains
# <br> 7. If the daily returns are between -3 and -7 that means top losses
# <br> 8. If the daily returns are greater than 7 that means **bull run** (stock prices are on rise)
# <br> 9. If the daily returns are lesser than  -7 that means **bear drop** (stock prices are on decline)
# <br><br> **Overview of bear and bull markets** : The words bull and bear market are used to explain how financial prices usually do — whether or not they increase or depreciate in value. At the same time , given that the economy is driven by the behavior of consumers, these words often reflect how investors feel about the business and the patterns that follow.

# In[103]:


df['Day_Perc_Change'] = df['Close Price'].pct_change(1)*100


# In[104]:


def change(df):
    if (df['Day_Perc_Change'] > -0.5) and (df['Day_Perc_Change'] < 0.5):
        return 'Slight or No Change'
    elif (df['Day_Perc_Change'] > 0.5) and (df['Day_Perc_Change'] < 1):
        return 'Slight positive'
    elif (df['Day_Perc_Change'] > -1) and (df['Day_Perc_Change'] < -0.5):
        return 'Slight negative'
    elif (df['Day_Perc_Change'] > 1) and (df['Day_Perc_Change'] < 3):
        return 'Positive'
    elif (df['Day_Perc_Change'] > -3) and (df['Day_Perc_Change'] < -1):
        return 'Negative'
    elif (df['Day_Perc_Change'] > 3) and (df['Day_Perc_Change'] < 7):
        return 'Among top gainers'
    elif (df['Day_Perc_Change'] > -7) and (df['Day_Perc_Change'] < -3):
        return 'Among top losers'
    elif (df['Day_Perc_Change'] > 7):
        return 'Bull run'
    elif (df['Day_Perc_Change'] < -7):
        return 'Bear drop'

df['Trend'] = df.apply(change, axis = 1)


# In[105]:


df.head()


# In[106]:


df['Total Traded Quantity'].groupby(df.Trend, group_keys=False).mean()


# In[107]:


df['Total Traded Quantity'].groupby(df.Trend, group_keys=False).median()


# In[108]:


df.index = df['Date']
df.sort_index(inplace = True)
del df['Date']


# In[109]:


df.head()


# In[110]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")


# In[111]:


plt.figure(figsize=(20, 10))
df['Close Price'].plot(kind = 'line')
plt.title('Close Price')
plt.ylabel('Closing Price')
plt.xlabel('Date')
plt.tight_layout()
plt.show()


# The above graph plots the closing price for the entire time frame of the stock. It displays the when the closing price of the stock was the highest which is around August 2018 and when the closing price was the lowest which is around December 2018. It gives an approximate idea about how the closing the price of the stock is varying over the time.

# In[112]:


plt.figure(figsize=(30, 10))
plt.stem(df.index, df['Day_Perc_Change'], use_line_collection=True)
plt.show()


# The above graph is a stem plot which is a discrete series plot. Stem plot is kind of similar to histogram. It helps to gain insights from the dataset and highlighting the outliers in the dataset. The daily returns of the closing price of the stock are plotted over the time.  

# In[113]:


scaledvolume =  df["Total Traded Quantity"] - df["Total Traded Quantity"].min()
scaledvolume = scaledvolume/scaledvolume.max() * df.Day_Perc_Change.max()

fig, ax = plt.subplots(figsize=(12, 6))

ax.stem(df.index, df.Day_Perc_Change , 'b', markerfmt='bo', label='Percentage Change', use_line_collection = True)
ax.plot(df.index, scaledvolume, 'k', label='Volume')

ax.set_xlabel('Date')
plt.legend(loc=2)

plt.tight_layout()
plt.xticks(plt.xticks()[0], df.index.date, rotation=45)
plt.show()


# The above graph plots the total traded quantity of the stock for the entire timeframe over the stem plot of daily returns. It helps to derive a relation between the daily returns of the stocks and the total traded quantity. As it is visible in the plot, the total traded quantity usually increases when the daily returns are high indicating a positive relationship but it not always true in some parts.

# In[114]:


df['Trend'].value_counts()


# In[115]:


plt.figure(figsize=(30, 10))
df['Trend'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', startangle=90)
plt.tight_layout()
plt.show()


# The above plot is a pie chart that shows the percentage of each type of trend in the stocks. The trend constituting the highest percentage is the slight or no change in the daily returns of the stock indicating that the stock is non volatile. 

# In[116]:


df['Total Traded Quantity'].groupby(df.Trend, group_keys=False).mean().plot(kind='bar', color = 'blue', label='mean')
plt.title('mean')
plt.show()


# In[117]:


df['Total Traded Quantity'].groupby(df.Trend, group_keys=False).median().plot(kind='bar', color='red', label='median')
plt.title('median')
plt.show()


# The above plots are bar graphs. It plots the mean and median respectively of the total traded quantity of the stock grouped by the trend it follows.

# In[118]:


mini = df['Day_Perc_Change'].min().round().astype('int64')
maxi = df['Day_Perc_Change'].max().round().astype('int64')
df['Day_Perc_Change'].plot(kind = 'hist', bins=range(mini, maxi), color='lightblue')
plt.show()


# The above plot is a histogram which plots the daily returns of the stock.

# In[119]:


result = df['Day_Perc_Change'].rolling(7).std()
nifty = pd.read_csv('Nifty50.csv')
nifty.Date = nifty.Date.astype('datetime64')
nifty = nifty.set_index('Date')
nifty['Perc_Change'] = nifty['Close'].pct_change()*100
result2 = nifty['Perc_Change'].rolling(7).std()


# In[120]:


x = df.index
y = np.array(result)
x2 = nifty.index
y2 = np.array(result2)
plt.figure(figsize=(25, 10))
plt.plot(x, y, label='Google')
plt.plot(x2, y2, label='Nifty')
plt.ylabel('Volatility')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()


# The above plot is volatility vs time plot.
# <br> **Volatility** is the change in variance in the returns of the stock over a specific period of time. If the stock market falls or rises more than 1% over a sustained period of time it is called a volatile market. 
# <br> To analyze the volatility of a stock, a general trend of the market is needed. The **NIFTY 50** index is National Stock Exchange of India's benchmark broad based stock market index for the Indian equity market. NIFTY 50 stands for National Index Fifty, and represents the weighted average of 50 Indian company stocks in 17 sectors. NIFTY 50 provides the general trend of the stock market along which the volatility of a stock is analyzed. 
# <br> To calculate the Volatility : First the 7 day rolling average also called moving average of the daily returns is calculated using rolling function provided by the pandas library. Then the standard deviation of the whole is calculated. 
# <br> If a stock is volatile, it means it is riskier to invest in them but one can expect high returns from them.

# In[121]:


plt.figure(figsize=(25, 10))
plt.plot(df.index, df['Close Price'].rolling(21).mean(), label = '21_SMA', color='blue')
plt.plot(df.index, df['Close Price'].rolling(34).mean(), label = '34_SMA', color='green')
plt.ylabel('Price')
plt.xlabel('Date')
plt.tight_layout()
plt.show()


# The above plot plots the 21 day moving average and 34 day moving average. The rolling mean of the closing price is calculated for 21 and 34 days respectively. 
# <br> It is a common strategy used in finance to make the call. If the smaller moving average that is the 21 moving average crosses over the longer moving average i.e. the 34 moving average then the call is to **buy** the stock otherwise the call is to **sell** the stock.

# In[122]:


rm = df['Close Price'].rolling(14).mean()
rstd = df['Close Price'].rolling(14).std()
upper_band = rm + (rstd * 2)
lower_band = rm - (rstd * 2)
plt.figure(figsize=(25, 10))
plt.plot(df.index, rm, color='green')
plt.plot(df.index, upper_band, color='blue')
plt.plot(df.index, lower_band, color='red')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()


# The above plot is of Bollinger Bands. **Bollinger Bands** are a type of statistical chart characterizing the prices and volatility over time.
# <br> It consists of 3 bands :- 1. **14 day rolling/moving average (green band)**
# <br> 2. **Upper Band (blue band)** - Calculated using the addition of 2 times the standard deviation of the 14 day rolling standard deviation from the 14 day rolling mean.
# <br> 3. **Lower Band (red band)** - Calculated using the subtraction of 2 times the standard deviation of the 14 day rolling standard deviation from the 14 day rolling mean.
# <br> If the 14 day rolling average is closer to upper band it indicates that the stock is **overbought** whereas if it closer to lower band it indicates that the stock is **oversold**.

# In[123]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm 
from statsmodels.regression.linear_model import OLS


# In this part, the OHLC prices of gold along with total volume traded is loaded. It contains two columns - Pred and New where Pred column is incomplete. The dataset is divided into two parts - training and testing. Using linear regression, rest of the values of the pred column are predicted by fitting the model to the training set. The independent variables are the OHLC prices. Similarly, using Polynomial regression, the model is fitted to the training set for the target variable new. 

# In[124]:


gold = pd.read_csv('GOLD.csv')
gold.head()


# In[125]:


train = gold.iloc[0:411,:]
x_train, y_train = train.loc[:,'Price':'Low'], train['Pred']
test = gold.iloc[411:512,:]
x_test, y_test = test.loc[:,'Price':'Low'], test['Pred']


# In[126]:


reg =LinearRegression().fit(x_train, y_train)


# In[127]:


reg.coef_


# In[128]:


reg.score(x_train, y_train)


# In[129]:


y_test = reg.predict(x_test)
y_test = pd.Series(y_test)
pred = pd.concat([y_train, y_test], ignore_index=True)
df['Pred'] = pred


# In[130]:


gold.iloc[0:411,:]


# In[131]:


gold.iloc[411:512,:]


# In[132]:


new_x = gold.loc[:, 'Price':'Low']
new_y = gold['new']
new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size = 0.2)


# In[133]:


new_reg =LinearRegression().fit(new_x_train, new_y_train )


# In[134]:


new_reg.coef_


# In[135]:


new_pred = new_reg.predict(new_x_test)
new_pred


# In[136]:


mse = mean_squared_error(new_y_test, new_pred)
mse


# In[137]:


r2 = new_reg.score(new_x_train, new_y_train)
r2


# In[138]:


poly = PolynomialFeatures(2)
new_x_train_poly = poly.fit_transform(new_x_train)
poly_reg = LinearRegression().fit(new_x_train_poly, new_y_train)
improved_r2 = poly_reg.score(new_x_train_poly, new_y_train)
improved_r2


# In this part, the OLS regression is applied on the google and nifty 50 stocks. In statistics, **ordinary least squares (OLS)** is a type of linear least squares method for estimating the unknown parameters in a linear regression model. It gives the beta value of a stock. **Beta value** of an asset is a measure of sensitivity of its returns relative to a market benchmark (nifty 50). High beta corresponds to high risk but ofter high returns. 

# In[140]:


prices = pd.concat([df['Close Price'], nifty['Close']], axis = 1)
prices.columns = ['Google', 'Nifty50']
daily_returns = prices.pct_change().dropna(axis=0)
daily_returns.head()


# In[141]:


x = daily_returns['Nifty50']
y = daily_returns['Google']
x1 = sm.add_constant(x)

model = OLS(y, x1)
model.fit().summary()


# Daily Beta value for past 3 months = 0.2857
# <br> The beta values are less than 1 and are low that means that stocks of google are less risky. The price is steadier as compared to other stocks in the market. 
# <br> If the Beta values were negative, it would indicate an inverse relation to the market that is they would do good if general trend of market falls.

# In[143]:


from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[144]:


df['Volatility'] = df['Close Price'].pct_change().rolling(7).std()
df.tail()


# In[145]:


df["14 day SMA"] = df['Close Price'].rolling(14).mean()
df["14 day STD"] = df['Close Price'].rolling(14).std()
df["bollinger upper band"] = df["14 day SMA"] + df["14 day STD"]*2
df["bollinger lower band"] = df["14 day SMA"] - df["14 day STD"]*2
df.tail()


# The below call function makes a call based on the bollinger bands and closing price of the stock.
# <br> If the closing price is less than or equal to the lower bollinger band, then the call is to **buy**
# <br> If the closing price is more than or equal to the upper bollinger band, then the call is to **short**
# <br> If the closing price is between the lower bollinger band and 14 moving average, then the call is to **Hold Buy/Liquidate Short**
# <br> If the closing price is between the upper bollinger band and 14 moving average, then the call is to **Hold Short/Liquidate Buy**

# In[146]:


def call(df): 
    if df['Close Price'] <= df["bollinger lower band"]:
        return 'Buy'
    elif df['Close Price'] >= df["bollinger upper band"]:
        return 'Short'
    elif df['Close Price'] > df["bollinger lower band"] and df['Close Price'] < df['14 day SMA']:
        return 'Hold Buy/ Liquidate Short'
    elif df['Close Price'] < df["bollinger upper band"] and df['Close Price'] > df['14 day SMA']:
        return 'Hold Short/ Liquidate Buy'
df['Call'] =  df.apply(call, axis = 1)
df['Call'].value_counts()


# Based on the 4 parameters i.e. Closing price and 3 bollinger bands - a classification model is trained to predict the call the future stocks.

# In[147]:


scaler = StandardScaler()
X = df[['Close Price', '14 day SMA', 'bollinger lower band', 'bollinger upper band']]
Y = df['Call'].dropna()
X = X.dropna()
X = scaler.fit_transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 0)
features = len(df['Call'].unique())


# In[148]:


classifiers = OrderedDict([
    ( "KNN", KNeighborsClassifier(features) ),
    ( "SVM",        SVC(kernel="linear", C=0.025) ),
    ( "Decision Tree",     DecisionTreeClassifier(max_depth=5) ),
    ( "Neural Net",        MLPClassifier(alpha=1, max_iter=1000) ),
    ( "Naive Bayes",       GaussianNB() ),
])


# In[149]:


r2scores = {}
for (name, classifier) in classifiers.items():
  classifier.fit(xtrain, ytrain)
  r2scores[name] = classifier.score(xtest, ytest)
    
r2scores = OrderedDict(sorted(r2scores.items(), key=lambda x: x[1]))
r2scores


# As the Neural Net gives the highest r2 score i.e. highest accurate prediction, it is selected as the classification model to predict the call for following TCS stock.

# In[150]:


Tcs = pd.read_csv('TCS.csv')
Tcs.head()


# In[151]:


Tcs['Date'] = Tcs['Date'].astype('datetime64')
Tcs = Tcs[Tcs.Series == 'EQ']
Tcs.set_index('Date', inplace = True)


# In[152]:


Tcs["14 day SMA"] = Tcs['Close Price'].rolling(14).mean()
Tcs["14 day STD"] = Tcs['Close Price'].rolling(14).std()
Tcs["bollinger upper band"] = Tcs["14 day SMA"] + Tcs["14 day STD"]*2
Tcs["bollinger lower band"] = Tcs["14 day SMA"] - Tcs["14 day STD"]*2
Tcs = Tcs.dropna()


# In[153]:


Tcs_xtrain = Tcs[['Close Price', '14 day SMA', 'bollinger upper band', 'bollinger lower band']]
Tcs_xtrain = scaler.fit_transform(Tcs_xtrain)
Tcs['Call'] = classifiers['Neural Net'].predict(Tcs_xtrain)
Tcs


# In the google stock, the calls were made manually that is the conditions were hand-coded but in case of TCS stocks, the calls are predicted using the classification model, taking the same parameters of closing price and bollinger bands.

# In[160]:


plt.figure(figsize=(30, 10))
df['Call'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', startangle=90)
plt.tight_layout()
plt.title('Google Condition Based Calls')
plt.show()


# The above piechart shows the percentage of different calls for google stocks. Hold Short or Liquidate Buy holds the maximum percentage whereas the call to Buy has the least percentage in the stocks.

# In[158]:


plt.figure(figsize=(30, 10))
Tcs['Call'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', startangle=90)
plt.tight_layout()
plt.title('TCS Predicted Calls')
plt.show()


# The above pie chart is of the calls that are made using the classification model. It is seen that the call to Hold Short or Liquidate Buy constitutes the highest percentage whereas the call to short constitutes the lowest percentage. 
# <br> In case of the stocks of Google the least percentage was the call to buy whereas in the stocks of TCS the least percentage is the call to short.
