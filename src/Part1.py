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


df= pd.read_csv('cowin_vaccine_data_statewise1.csv') 
df

df.info()

df['Total Doses Administered'].plot()

df['Total Individuals Vaccinated'].plot()


df1=df.loc[df['State'] == "Delhi"]
df1

# df1.drop(['Total Sessions Conducted','First Dose Administered','Second Dose Administered','Male(Individuals Vaccinated)','Female(Individuals Vaccinated)','Transgender(Individuals Vaccinated)'], axis=1, inplace=True)

# df1.drop(['Total Covaxin Administered','Total CoviShield Administered','AEFI','18-30 years (Age)','30-45 years (Age)','45-60 years (Age)','60+ years (Age)'], axis=1, inplace=True)

# df1

df2=df.loc[df['State'] == "Maharashtra"]
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3=df.loc[df['State'] == "Uttar Pradesh"]
df3.reset_index(drop=True, inplace=True)
df4=df.loc[df['State'] == "Rajasthan"]
df4.reset_index(drop=True, inplace=True)
df2.head()
df3.head()
df4.head()

df1['Total Individuals Vaccinated'].plot()
df2['Total Individuals Vaccinated'].plot()
df3['Total Individuals Vaccinated'].plot()
df4['Total Individuals Vaccinated'].plot()

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

df["Updated On"]= pd.to_datetime(df["Updated On"])

plt.figure(figsize = (10,7))
plt.plot( df1['Total Individuals Vaccinated'], label='Total Individuals Vaccinated DELHI')
plt.plot( df2['Total Individuals Vaccinated'], label='Total Individuals Vaccinated MAHARASHTA')
plt.plot( df3['Total Individuals Vaccinated'], label='Total Individuals Vaccinated U.P.')
plt.plot( df4['Total Individuals Vaccinated'], label='Total Individuals Vaccinated RAJASTHAN')
plt.xlabel("Dates in Number")
plt.ylabel("Number of Individuals Vaccinated")
plt.title("Total Vaccination doses")
# # myFmt = mdates.DateFormatter("%Y-%m-%d")
# # plt.gca().xaxis.set_major_formatter(myFmt)
# plt.xaxis.set_major_locator(mdates.WeekdayLocator(interval=50))
# plt.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y"))
plt.legend()
plt.show()

data = df1['Total Individuals Vaccinated']
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
  testPredict = model.predict(X_test)
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

  #print(classification_report(y_test, testPredict))
  #print(confusion_matrix(y_test, testPredict))

  #MNB_f1 = round(f1_score(y_test, testPredict, average='weighted'), 3)
  #MNB_accuracy = round((accuracy_score(y_test, testPredict)*100),2)
  score=model.evaluate(X_train, Y_train)
  # print("Accuracy : " , round((score[1]*100),2), " %")
  #print("f1_score : " , MNB_f1)

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
plt.xlabel('Time in days')
plt.ylabel('Daily vaccinated individuals in MADHYA PRADESH')
plt.title("Prediction of number of vaccinations required in Madhya Pradesh")
plt.grid(True)
plt.show()

# %%capture
# !wget -nc https://raw.githubusercontent.com/brpy/colab-pdf/master/colab_pdf.py
# from colab_pdf import colab_pdf
# colab_pdf('Untitled14.ipynb')