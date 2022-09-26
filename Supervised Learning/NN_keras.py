import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential,save_model, load_model
from keras.layers import Dense, Activation
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.layers.core import Dropout, Dense
#import pandas as pd
#df = pd.read_csv("/Users/jojomoolayil/Book/Ch3/Data/train.csv")
from sklearn.preprocessing import StandardScaler

def NN(X_train, Y_train, X_val, Y_val, X_test, Y_test):

    history = History()
    columns_X = np.size(X_train[0,:])  
    columns_Y = np.size(Y_train[0,:]) 

    # scaler = StandardScaler()
    # scaler.fit(Y_train)
    # y_train_scaled = scaler.transform(Y_train)
    # y_val_scaled = scaler.transform(Y_val)
    # y_test_scaled = scaler.transform(Y_test)

    y_train_scaled = Y_train
    y_val_scaled = Y_val
    y_test_scaled = Y_test
 
    #Activation functions
    # relu function             0.56
    # sigmoid function          0.55
    # softmax function
    # softplus function
    # softsign function
    # tanh function
    # selu function
    # elu function
    # exponential function 

    #Define the model architecture
    model = Sequential()
    model.add(Dense(32, input_dim=columns_X , activation = "relu")) #Layer 1
    #model.add(Dropout(0.2))
    model.add(Dense(16, input_dim=columns_X , activation = "relu")) #Layer 1
    # #model.add(Dropout(0.2))
    model.add(Dense(16, input_dim=columns_X , activation = "relu")) #Layer 1
    # # model.add(Dropout(0.2))

    model.add(Dense(columns_Y)) #OutputLayer

    #Configure the model
    opt = keras.optimizers.Adam(learning_rate=0.05)#, momentum = 0.4)
    model.compile(optimizer = opt ,loss="mean_squared_error",metrics=['mean_absolute_error'])

    #Train the model
    model.fit(X_train, y_train_scaled, batch_size=64, epochs=500, validation_data=(X_val,y_val_scaled), callbacks=[history])

    #Evaluate with test data
    result = model.evaluate(X_test,y_test_scaled)

    for i in range(len(model.metrics_names)):
        print("Metric ",model.metrics_names[i],":",str(round(result[i],2)))

    print("Mean for Train Data:",abs( Y_train.mean()).mean())

    #Manually predicting from the model, instead of using model's evaluate function
    Y = model.predict(X_train)
    Y_ = model.predict(X_val)

    # print(y_train_scaled[:,:])
    # print(Y[:,:])
    plt.title("Cp vs Espesor")
    plt.ylabel('cp')
    plt.xlabel('Espesor')
    plt.plot(X_train[13,0:int(columns_X/2):1], y_train_scaled[13,:])
    plt.plot(X_train[13,0:int(columns_X/2):1], Y[13,:])
    plt.legend(['Test Data', 'Prediction'], loc='upper left')
    plt.show()

    plt.title("Cp vs Curvatura")
    plt.ylabel('cp')
    plt.xlabel('Curvatura')
    plt.plot(X_train[13,int(columns_X/2):int(columns_X-1):1], y_train_scaled[13,:])
    plt.plot(X_train[13,int(columns_X/2):int(columns_X-1):1], Y[13,:])
    plt.legend(['Test Data', 'Prediction'], loc='upper left')
    plt.show()

    cuerda = np.zeros((int(columns_X/2)), dtype=np.double)
    for i in range(0, int(columns_X/2)):
        cuerda[i] = 2 - 4*(i/int(columns_X/2))

    plt.title("Cp vs cuerda")
    plt.ylabel('cp')
    plt.xlabel('cuerda')
    plt.plot(cuerda, y_train_scaled[13,:])
    plt.plot(cuerda, Y[13,:])
    plt.legend(['Test Data', 'Prediction'], loc='upper right')
    plt.show()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model's Training & Validation loss across epochs")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Validation', 'Train'], loc='upper right')
    plt.show()

    plt.hist(abs(Y[:,0] - Y_train[:,0]))
    plt.show() 

    
    plt.hist(abs(Y[:,24] - Y_train[:,24]))
    plt.show() 

    plt.hist(abs(Y - Y_train).flatten())
    plt.show() 

    #Save model
    filepath = "./saved_airfoil_model"
    save_model(model, filepath)

    return 

