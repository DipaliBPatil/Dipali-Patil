
## Kaggle challenge on Bike Share demand ##

## About the Data :
 Hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month.We must have to predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.
 
## Data Fields

datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals



```python
import numpy as np 
import pandas as pd
df=pd.read_csv("D:\\train.csv",parse_dates=['datetime'])

```


```python
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday']=df['datetime'].dt.weekday

df['season'] = df['season'].astype('category')
df['weather'] = df['weather'].astype('category')
df.drop(columns=['datetime','casual', 'registered'],inplace=True)
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.drop('count',axis=1)
y = dum_df['count']
X.head()


y.head()





```




    0    16
    1    40
    2    32
    3    13
    4     1
    Name: count, dtype: int64




```python
df_test = pd.read_csv("D:\\test.csv",parse_dates=['datetime'])
df_test['year'] = df_test['datetime'].dt.year
df_test['month'] = df_test['datetime'].dt.month
df_test['day'] = df_test['datetime'].dt.day
df_test['hour'] = df_test['datetime'].dt.hour
df_test['weekday']=df_test['datetime'].dt.weekday

df_test['season'] = df_test['season'].astype('category')
df_test['weather'] = df_test['weather'].astype('category')
df_test.drop(columns=['datetime'],inplace=True)
dum_df_test = pd.get_dummies(df_test, drop_first=True)


from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=2019,
                                  n_estimators=500,oob_score=True)
model_rf.fit( X,y )
y_pred = model_rf.predict(dum_df_test)
y_pred[y_pred<0] = 0
y_pred


```




    array([ 13.966,   6.358,   5.496, ..., 166.032, 118.034,  75.308])




```python
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_log_error
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

model_rf.fit( X_train , y_train )
y_pred2 = model_rf.predict(X_test)
y_pred2[y_pred2<0] = 0
print("Root Mean Squared Log Error  = %6.4f" % 
      np.sqrt(mean_squared_log_error(y_test, y_pred2)))
```

    Root Mean Squared Log Error  = 0.3412
    


```python

```
