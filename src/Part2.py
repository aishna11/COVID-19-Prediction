import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU, Bidirectional
from keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# %matplotlib inline 
import matplotlib
import matplotlib.pyplot as plt 

data_in= pd.read_csv('case_time_series (4).csv')
data_in

data_in["Date"]= pd.to_datetime(data_in["Date"]) 
data_in

data_in.info()

data_in.tail()

data_in.drop(['Total Confirmed','Total Recovered','Total Deceased'], axis=1, inplace=True)

data_in.info()

data_in

df1  = data_in[data_in['Date']>'2020-03-18']
# df1.drop(['location'],axis = 1,inplace = True)
df1.reset_index(drop=True, inplace=True)
df1

# df1.drop(['total_cases','total_deaths'],axis =1,inplace = True)

data = df1['Daily Confirmed']

df1.to_csv('case_time_series (4).csv')

df1['Daily Confirmed'].plot()

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

df1["Date_YMD"]= pd.to_datetime(df1["Date_YMD"])

plt.figure(figsize = (15,10))
plt.plot(df1['Date_YMD'], df1['Daily Confirmed'], label='Daily Confirmed')
# plt.plot(df1['Date_YMD'], df1['Daily Recovered'], label='Daily Recovered')
# plt.plot(df1['Date_YMD'], df1['Daily Deceased'], label='Daily Deceased')
plt.xlabel("Dates in Number")
plt.ylabel("Number of Cases")
plt.title("Daily Death Cases for covid")
myFmt = mdates.DateFormatter("%d-%m-%Y")
plt.gca().xaxis.set_major_formatter(myFmt)
plt.legend()
plt.show()

df2 = pd.read_csv('case_time_series (4).csv')
df2.head()

data = df2['Daily Confirmed']


look_back = 14
data = np.array(data)
data = data.reshape(len(data),1)
print(data.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
train, test= np.split(data, [int(.8 *len(data))])
train = train.reshape(len(train) , 1)
test = test.reshape(len(test) , 1)
print(test.shape)
print(train.shape)
def preprocessing(data , days=look_back):
    X, Y = [], []
    for i in range(len(data)-days-1):
        a = data[i:(i+days), 0]
        X.append(a)
        Y.append(data[i + days, 0])
    return np.array(X), np.array(Y)

X_train,Y_train =preprocessing(train)
X_test,Y_test =preprocessing(test)
Y_train = Y_train.reshape(len(Y_train),1)
Y_test = Y_test.reshape(len(Y_test),1)
X_train = X_train.reshape(X_train.shape[0] , 1 ,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0] , 1 ,X_test.shape[1])
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
X,Y = preprocessing(data)
Y = Y.reshape(len(Y),1)
X = X.reshape(X.shape[0] , 1 ,X.shape[1])
def learning_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return

def model_score(model, X_train, y_train, X_test, y_test):

    # trainPredict = model.predict(X_train)
    # testPredict = model.predict(X_test)
    # trainPredict = scaler.inverse_transform(trainPredict)
    # y_train = scaler.inverse_transform(y_train)
    # testPredict = scaler.inverse_transform(testPredict)
    # y_test = scaler.inverse_transform(y_test)

    # print(y_train.shape)
    # print(trainPredict.shape)

    # trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[0]))
    # print('Test Score: %.2f RMSE' % (testScore))
    # return trainScore, testScore

    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.5f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.5f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

days = 14
model = Sequential()
model.add(GRU(256 , input_shape = (1 , days) , return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(64 ,  activation = 'relu'))
model.add(Dense(1))
print(model.summary())
optimizer = optimizers.Adam(lr=0.01)

model.compile(loss='mean_squared_error', optimizer=optimizer , metrics = ['mean_squared_error'])
history = model.fit(X_train, Y_train, epochs=100 , batch_size = 128 , validation_data = (X_test,Y_test))
learning_plot(history)

model_score(model, X_train, Y_train , X_test, Y_test)

pred = model.predict(X_train)
pred = scaler.inverse_transform(pred)
y_test = Y_train.reshape(Y_train.shape[0] , 1)
y_test = scaler.inverse_transform(y_test)
print("Red - Predicted,  Blue - Actual")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(y_test , 'b')
plt.plot(pred , 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.grid(True)
plt.show()

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def model_graph(X,model,Y):
    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)
    y_test = Y.reshape(Y.shape[0] , 1)
    y_test = scaler.inverse_transform(y_test)
    print("Red - Predicted,  Blue - Actual")
    plt.rcParams["figure.figsize"] = (15,7)
    plt.plot(y_test , 'b')
    plt.plot(pred , 'r')
    plt.axvline(train.shape[0], color='g') 
    plt.xlabel('Time in days')
    plt.ylabel('Daily Death Cases')
    # myFmt = mdates.DateFormatter("%d-%m-%Y")
    # plt.gca().xaxis.set_major_formatter(myFmt)
    plt.title("stacked LSTM for Daily Death cases prediction")
    plt.grid(True)
    plt.show()
    return
model_graph(X,model,Y)

model_b = Sequential()
model_b.add(Bidirectional(LSTM(256, activation='relu',return_sequences = True), input_shape=(1,14)))
model_b.add(Dropout(0.3))
model_b.add(Dense(128 ,  activation = 'relu'))

model_b.add(Dense(1,activation='relu'))

print(model_b.summary())

optimizer = optimizers.Adam(lr=0.001)
model_b.compile(loss='mean_squared_error', optimizer=optimizer , metrics = ['mean_squared_error'])
history_b = model_b.fit(X_train, Y_train, epochs=100 , batch_size = 32 , validation_data = (X_test,Y_test))

plt.plot(history_b.history['loss'])
plt.plot(history_b.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model_score(model_b, X_train, Y_train , X_test, Y_test)

pred = model_b.predict(X)
pred1 = model.predict(X)
pred1 = scaler.inverse_transform(pred1)
y_test1 = Y.reshape(Y.shape[0] , 1)
y_test1 = scaler.inverse_transform(y_test1)
pred = pred.reshape(pred.shape[0],1)
pred = scaler.inverse_transform(pred)
y_test = Y.reshape(Y.shape[0] , 1)
y_test = scaler.inverse_transform(y_test)
print("Blue - Actual , Red -Bidirectional Predicted , Yellow- Stacked Prediction")
plt.rcParams["figure.figsize"] = (15,10)
plt.plot(y_test , 'b')
plt.plot(pred , 'r')
plt.plot(pred1 , 'y')
plt.xlabel('Time in days')
plt.axvline(train.shape[0], color='g')
plt.ylabel('Daily new Cases')
plt.title("Daily Death prediction for Covid-19 in India")
plt.grid(True)
plt.show()

close_data = X.reshape((-1))
look_back = 14

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x.reshape((1,1,14)))[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df2['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30
forecast_dates = predict_dates(num_prediction)
forecast = predict(num_prediction, model_b).reshape(-1,1)
forecast1 = scaler.inverse_transform(forecast)

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

pred = model_b.predict(X)
pred = pred.reshape(pred.shape[0],1)
pred = np.concatenate((pred, forecast), axis = 0)
pred = scaler.inverse_transform(pred)
# Yforecast = np.concatenate((Y, forecast), axis = 0)
y_test = Y.reshape(Y.shape[0] , 1)
y_test = scaler.inverse_transform(y_test)
print("Red - Predicted,  Blue - Actual")
plt.rcParams["figure.figsize"] = (15,10)
plt.plot(y_test , 'b')
plt.plot(pred , 'r')
plt.xlabel('Time in days')
plt.axvline(train.shape[0], color='c')
plt.axvline(Y.shape[0], color='g')
# plt.axvline(x=53, color='y', label='lockdown 1-23/03/2020')
# plt.axvline(x=76, label='lockdown 2-15/04/2020', color='y')
# plt.axvline(x=95, label='lockdown 3-04/05/2020', color='y')
# plt.axvline(x=109, label='lockdown 4-18/05/2020', color='y')
# plt.axvline(x=123, label='Unlock 1-01/06/2020', color='y')
# plt.axvline(x=153, label='Unlock 2-01/07/2020', color='y')
# plt.axvline(x=184, label='Unlock 3-01/08/2020', color='y')
# myFmt = mdates.DateFormatter("%d-%m-%Y")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
plt.ylabel('Daily new Cases')
plt.title("Forecasted number of Confirmed cases")
plt.grid(True)
plt.show()


