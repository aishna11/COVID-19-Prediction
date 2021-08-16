# importing all imp libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
np.random.seed(7)

# !pip install plotly>=4.9.0
# !wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O /usr/local/bin/orca
# !chmod +x /usr/local/bin/orca
# !apt-get install xvfb libgtk2.0-0 libgconf-2-4

# data is of netflix from date-(1-aug-2003)_to_(28-aug-2020) from yahoo finance
df = pd.read_csv('case_time_series (4).csv', header=0)
df = df.sort_index(ascending=True, axis=0)
df

df.drop(['Total Confirmed','Total Recovered','Total Deceased'], axis=1, inplace=True)

df.head()

df  = df[df['Date_YMD']>'2020-03-18']
df.reset_index(drop=True, inplace=True)
df.head()


df.drop(['Date'], axis=1, inplace=True)
df

df.shape

df.describe()

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

df["Date_YMD"]= pd.to_datetime(df["Date_YMD"])

plt.figure(figsize = (15,10))
plt.plot(df['Date_YMD'], df['Daily Confirmed'], label='Daily Confirmed')
plt.plot(df['Date_YMD'], df['Daily Recovered'], label='Daily Recovered')
plt.plot(df['Date_YMD'], df['Daily Deceased'], label='Daily Deceased')
plt.xlabel("Dates in Number")
plt.ylabel("Number of Cases")
plt.title("Daily Cases for covid")
myFmt = mdates.DateFormatter("%d-%m-%Y")
plt.gca().xaxis.set_major_formatter(myFmt)
plt.legend()
plt.show()

df.isnull().any()

fig = px.line(df, x='Date', y='Daily Confirmed')
fig.show()

fig = px.line(df, x=df['Date'], y='Daily Confirmed',title='cases in India')
fig.add_scatter(x=df['Date'], y=df['Daily Recovered'],mode='lines')
fig.show()

# pip install -U kaleido

if not os.path.exists("images"):
    os.mkdir("images")

crd_data = df.iloc[:, 2:5]
crd_avg = crd_data.mean(axis=1)
cr_avg = df[['Daily Confirmed', 'Daily Recovered']].mean(axis=1)
close = df['Daily Deceased']

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = df.index, y = crd_avg,name='CRD avg'))
fig1.add_trace(go.Scatter(x = df.index, y = cr_avg,name='CR avg'))
fig1.add_trace(go.Scatter(x = df.index, y = close,name='deaths column data'))
fig1.show()

fig1.write_image("diff_btwn_diff_avgs.png")

# we will create a new df which has only 2 column which is useful to predict data
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'crd_avg'])
for i in range(0, len(df)):
  new_data['Date'][i] = df['Date'][i]
  new_data['crd_avg'][i] = cr_avg[i]

new_data.head()

# setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

print(len(new_data))

ds = new_data.values

train.shape

test.shape

# we have normalize the data cuz data is like 149...., 488..something like that
# so we have to normalize betwwen 0 and 1
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(ds)

# splitting the data to x_train, y_train
# we will first train upto 60 and then predict on 61 and then 
# we will train from 61 to 120 then predict on 121 likewise we will go
x_train, y_train = [], []
for i in range(60, len(train)):
  x_train.append(scaled_data[i-60:i,0])
  y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# now we have reshape the array to 3-d to pass the data into lstm [number of samples, time steps/batch_size, features] 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train.shape

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.25))
model.add(LSTM(units=50))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

# predicting 920 values, using past 60 from the train data
inputs = new_data[len(new_data)-len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scalar.transform(inputs)

inputs.shape

x_test = []
for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price = model.predict(x_test)
# inverse transform for getting back all normal values from scaled values
predicted_price = scalar.inverse_transform(predicted_price)

rms=np.sqrt(np.mean(np.power((test-predicted_price),2)))
rms

test.tail()

# Graph for comparing the results of model predicted and original value
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = train.index, y = train.crd_avg,name='train'))
fig2.add_trace(go.Scatter(x = test.index, y = test.crd_avg,name='test_crd_avg'))
fig2.add_trace(go.Scatter(x = test.index, y = test.Prediction,name='test'))
fig2.show()


# %%capture
# !wget -nc https://raw.githubusercontent.com/brpy/colab-pdf/master/colab_pdf.py
# from colab_pdf import colab_pdf
# colab_pdf('Untitled33.ipynb')