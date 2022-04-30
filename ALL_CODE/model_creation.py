    


import numpy as np 
import math
import pandas as pd 
import pandas_datareader as pdr 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import sequential
from keras.layers import Dense , LSTM
from keras.models import Sequential
plt.style.use("fivethirtyeight")
import datetime
from datetime import  timedelta
        


#'NTPC.NS','HEROMOTOCO.NS','TECHM.NS'	,'COALINDIA.NS','APOLLOHOSP.NS','ITC.NS',
set = [
'HINDALCO.NS',	
'KOTAKBANK.NS',	
'LT.NS',	
'INDUSINDBK.NS'	,
'TATASTEEL.NS',	
'ONGC.NS'	,
'BRITANNIA.NS',	
'WIPRO.NS'	,
'CIPLA.NS'	,
'TITAN.NS'	,
'BAJAJFINSV.NS'	,
'ICICIBANK.NS'	,
'BAJAJ-AUTO.NS'	,
'NESTLEIND.NS'	,
'BHARTIARTL.NS'	,
'HDFCLIFE.NS'	,
'TATACONSUM.NS'	,
'TCS.NS'	,
'RELIANCE.NS',	
'MARUTI.NS'	,
'BAJFINANCE.NS',	
'ULTRACEMCO.NS'	,
'SHREECEM.NS'	,
'MM.NS']
for i in range(len(set)):




    # Get the stock quote
    # name should be exact from yahoo finance site
    stock_name = set[i]
    df = pdr.DataReader(stock_name , data_source='yahoo',start='2000-01-01',end=datetime.datetime.now())

    # show the data 
    df


    #df.to_csv('price.csv') to save data as csv

    df.shape

    # Creating a new df for only close price 
    #data = df.filter(['Close','High'])           # gives 2 column 
    data = df.filter(['Close'])                   # gives 1 column
    data

    # create df to a  num py array         df.values : Only the values in the DataFrame will be returned, the axes labels will be removed.

    dataset = data.values
    dataset


    # Get the no of adta to train the model on say 80%
    training_data_len = math.ceil(len(dataset)*0.80)                    #Return the ceiling of x as an Integral.
    training_data_len

    # scale the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)                        # range [0-1] on numpy array of dataset
    scaled_data




    # create a scaled training data set 
    train_data = scaled_data[0:training_data_len , : ]                # till row = training data len all column points

    # split the data into x_train  , and y_train data sets
    x_train = []                                                      # features = independent variables 
    y_train = []                                                      # lables = output = dependent variables 

    for i in range(60 , len(train_data)): 
        x_train.append(train_data[i-60:i , 0])                        # we are going to append 0 to 89th  values of train_data in 0th column
        y_train.append(train_data[i , 0])                             # lable/ prediction is 90th data point of 0th column from train_data to y_train
        
        if i <= 62:
            print(x_train)
            print()
            print(y_train)




    # convert x_train and y_train to numpy arrays so that array data can be provided to LSTM model
    x_train , y_train = np.array(x_train) , np.array(y_train)

    # x_train.shape               (2363, 30)   # after appending in every row tere are 30 columns of data 

    # x_train                                  # is 2d type row  and column data

    # reshaping the x_train data set as a LSTM model expects the input to be a three dimentionals array 
    x_train = np.reshape(x_train ,newshape= (x_train.shape[0] , x_train.shape[1] , 1))
    x_train.shape

    # Build the LSTM model 
    model = Sequential()

    # add a layer of 50 neuron , there output to be used in next layer thus return seq = true , since first layer thus shape of input neuron   , 
    # input shape = (time step , features ) = (90 days , 1 ) = (90 , close price)
    model.add(LSTM(50, activation ='relu', return_sequences=True, input_shape = (x_train.shape[1],1)))
    
    # add 1 last lstm layer but no lstm further thus return_sequence = False
    model.add(LSTM(50, return_sequences=False))

    # adding last dense layer          
    model.add(Dense(25))  

    # adding last dense layer          
    model.add(Dense(1)) 


    model.summary()



    # compile the model 
    # optimizer is used to minimise the loss fx 
    # loss fx is used to calculate what was the loss to see how well the model did on traiing 

    model.compile(optimizer='adam' , loss="mean_squared_error")




    # train the model
    model.fit(x=x_train,y=y_train,batch_size=1,epochs=10)



    model.save('{}_keras_model.h5'.format(stock_name))




    









