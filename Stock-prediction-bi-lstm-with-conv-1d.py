import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

from tensorflow import keras
from sklearn.metrics import mean_squared_error
import math

def return_rmse(test,predicted):
    
    rmse = math.sqrt(mean_squared_error(test, predicted))
    rmse=round(rmse,4)
    print("The root mean squared error is {}".format(rmse))

data=pd.read_csv('C:/Users/GIGABYTE/Downloads/Stock_Prediction_Bi-LSTM with Conv-1D/Database/INFY.csv',parse_dates=['Date'],index_col='Date')
data.head()

data['Day']=data.index.day
data['DayOfWeek']=data.index.dayofweek
data.head()


plt.figure(figsize=(30,15))
plt.plot(data.index,data['Close'])
plt.title('INFY Close price',fontsize=70)
plt.xlabel("Year",fontsize=40)
plt.ylabel("Close price",fontsize=40)
plt.savefig("C:/Users/GIGABYTE/Downloads/Stock_Prediction_Bi-LSTM with Conv-1D/img/INFY_Show_dataset.png")
plt.show()



train_size=int(len(data)*0.9)
train,test=data.iloc[:train_size],data.iloc[train_size:len(data)]
train.shape,test.shape


rs_data = MinMaxScaler()
rs_target = MinMaxScaler()

target=data['Close']
data.drop(columns=['Close'],inplace=True)

train.loc[:,data.columns]=rs_data.fit_transform(train.loc[:,data.columns].to_numpy())
train['Close']=rs_target.fit_transform(train[['Close']].to_numpy())
test.loc[:,data.columns]=rs_data.fit_transform(test.loc[:,data.columns].to_numpy())
test['Close']=rs_target.fit_transform(test[['Close']].to_numpy())


time_steps=5

x_train, y_train = create_dataset(train, train['Close'], time_steps)
x_test, y_test = create_dataset(test, test['Close'], time_steps)
x_train.shape,x_test.shape


model = keras.Sequential()
model.add(keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,return_sequences=True
    )
  ))
model.add(
  keras.layers.Bidirectional(
  keras.layers.LSTM(
  units=500,return_sequences=True
    )
  ))
model.add(
  keras.layers.Bidirectional(
  keras.layers.LSTM(units=500)
  )
)
model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Dense(units=100,activation='relu'))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(units=1))
model.compile(loss=keras.losses.Huber(),
              optimizer='adam',
              metrics=["mse"])

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.35,
    shuffle=False,
    verbose=1
)

pred=model.predict(x_test)
y_train_inv=rs_target.inverse_transform(y_train.reshape(1,-1))
y_test_inv=rs_target.inverse_transform(y_test.reshape(1,-1))
pred=rs_target.inverse_transform(pred.reshape(1,-1))

return_rmse(y_test_inv,pred)

plt.plot(y_test_inv.flatten(),marker='.',label='True')
plt.plot(pred.flatten(),'r',marker='.',label='Predicted')
plt.legend()
plt.title('INFY Close predict result|RMSE:0.7131',fontsize=20)
plt.savefig("C:/Users/GIGABYTE/Downloads/Stock_Prediction_Bi-LSTM with Conv-1D/img/INFY_Close_predict_result.png")
plt.show()

