#save the model and reload it again



import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model


df = pd.read_csv('data.csv')
X = df.drop('wage_per_hour', axis = 1).values
y = df['wage_per_hour'].values
#X_train,X_test,y_train,y_test = train_test_split()
early_stopping_monitor = EarlyStopping(patience = 10)

n_cols = X.shape[1]
model = Sequential()
model.add(Dense(10,activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X,y,epochs = 20, validation_split = 0.3,callbacks = [early_stopping_monitor])

model.save('model_file.h5py')

print("Loss function: " + model.loss)

df_test=pd.read_csv('test.csv')
X_test = df_test.drop('wage_per_hour', axis = 1).values
y_test = df_test['wage_per_hour'].values
predicted_wage = model.predict(X_test)
print("the predicted wage from the test data is : " + str(predicted_wage))
print("the difference in the predicted and actual value is : " + str(predicted_wage-y_test))

X_experience = df['experience_yrs'].values
print(X_experience)
plt.title('EDA of hourly wage prediction data for employees')
plt.plot(X_experience,y, "o")

#plt.legend()
plt.xlabel('Experience in years')
plt.ylabel('hourly wage')
plt.show(block = True)
plt.savefig('fig1.pdf')