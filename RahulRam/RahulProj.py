from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler

person = 'Rahul'
data = pd.read_csv(('{}Grades.csv'.format(person)))
train = []
labels = []
for x in range(data.shape[0]):
    date = list(data[['Year','Month','Day']].iloc[x])
    day = int((datetime.date(int(date[0]),int(date[1]),int(date[2])) - datetime.date(2019,9,4)).days)
    grade = int(data[['Grade (Decimal)']].iloc[x] * 100)
    total = int(data[['Point Total']].iloc[x])
    weight = int(data[['Weight (0.3/0.7)']].iloc[x] * 100)
    train.append(np.array([day, total, weight]))
    labels.append(grade)
traindf = pd.DataFrame(train)
traindf['labels'] = labels
traindf = traindf.sort_values(by=[0])
input_data = traindf[[1, 'labels']].reset_index(drop=True)

sc= MinMaxScaler(feature_range=(0,1))
input_data = sc.fit_transform(input_data)
x_train, y_train = [], []
lookback = 10
features = 2
for i in range(input_data.shape[0] - lookback - 1):
    t = []
    for j in range(0, lookback):
        t.append(input_data[(i + j)])
    x_train.append(t)
    y_train.append(input_data[i + lookback][1])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], lookback, features)
preds, future_preds = [], []
for x in range(10):
    model = Sequential()
    model.add(LSTM(50, input_shape=(lookback, features)))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train,y_train, batch_size=16, epochs=500)
    preds.append((model.predict(x_train)))
    future_preds.append(model.predict(input_data[-1:-(lookback+1):-1][::-1].reshape(1, lookback, features)))
preds, avg_future = np.array(preds), sum(np.array(future_preds))/10
avgpreds = [sum(preds[:,i])/10 for i in range(preds.shape[1])]
avgpreds = np.array(avgpreds)
avgpreds = list(avgpreds.reshape(avgpreds.shape[0]))
preddf = pd.DataFrame(y_train)
preddf['preds'] = avgpreds
print(avg_future)
preddf.to_csv(('{}preds.csv'.format(person)), index='False')
