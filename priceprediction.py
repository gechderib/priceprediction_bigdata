import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
a= dataset_train.head()
print(a)
# b = dataset_train.tail()
# c = dataset_train.info()
# print(c)
# Train Your Model
training_set = dataset_train.iloc[:,1:2].values

# Normalizing the Dataset.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)
print(scaled_training_set[1,0])

# # Creating X_train and y_train Data Structures.
x_train = []
y_train = []
for i in range (60, 1258):
    x_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

# print(x_train.shape)
# print(y_train.shape)

# # Reshape the Data.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape
print(x_train.shape)
# # Part 2 - Building
# provide high-level APIs used for easily building and training models

from keras.models import Sequential # The core idea of Sequential API is simply arranging the Keras layers in a sequential order
from keras.layers import Dense # 
from keras.layers import LSTM # LSTMs are predominately used to learn, process, and classify sequential data 
from keras.layers import Dropout # The function of the dropout layer is just to add noise so the model learns to generalize better

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))


# ###### Filter the model
regressor.compile(optimizer = "adam", loss="mean_squared_error")
regressor.fit(x_train, y_train, epochs=100,batch_size=32)

# # extract the acutal price at jun
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
actual_stock_price = dataset_test.iloc[:,1:2].values

# ##### prepering the input for model
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []

for i in range(60, 80):
  x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# ### predicting the value
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

###### plot the actual and predicted price for the data
plt.plot(actual_stock_price, color="red", label="Acutal Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stcok Price")
plt.legend()

print("working")

