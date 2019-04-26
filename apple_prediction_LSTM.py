def lstm_run():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import csv

    dataset_train = pd.read_csv('EOD-AAPL.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    print(dataset_train.head())

    df = dataset_train

    #setting index as date
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']

    #plot
    plt.figure(figsize=(16,8))
    plt.plot(df['Open'], label='Open Price history')
    plt.xlabel('Year')
    plt.ylabel('APPLE Stock Price')
    plt.show()

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    #print(training_set_scaled)

    X_train = []
    y_train = []
    for i in range(60, 9634):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    from keras.models import model_from_json
    #regressor=null
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    regressor = model_from_json(loaded_model_json)
    # load weights into new model
    regressor.load_weights("model.h5")
    print("Loaded model from disk")
    #loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    #regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    #print
    #(X_train,y_train)
    #import h5py
    #regressor.save("abcmodel.h5")

    dataset_test = pd.read_csv('EOD-AAPLtest.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values

    #print(real_stock_price)
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    #print(dataset_total)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    #print("ABC",inputs)
    #print(inputs,dataset_total)
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 81):
        X_test.append(inputs[i-60:i, 0])
    #print(X_test)
    X_test = np.array(X_test)
    #"print(X_test)"
    #print("X_test in dkjbjdkfekjf")
    #print(X_test.shape[0], X_test.shape[1], 1)
    #print(X_test)
    #print(X_train)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test[0][0])
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    #print(dataset_test['Open'])
    #print(predicted_stock_price)

    #rms=np.sqrt(np.mean(np.power((dataset_test['Open'].values-predicted_stock_price),2)))
    #print(rms)

    with open('comparison.csv','w', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(['LSTM'])
        for val in dataset_test['Open']:
            writer.writerow([val])

    plt.figure(figsize=(16,8))
    plt.plot(real_stock_price, color = 'black', label = 'APPLE Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted APPLE Stock Price')
    plt.title('APPLE Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('APPLE Stock Price')
    plt.legend()
    plt.show()

    
    #b = 0.01665003

    #"print(a)"
    #Date=266.0
    #Date=Date.reshape(1,-1)
    #a=X_test[len(X_test)-1]
    #a.pop(0)
    #a.append(sc.transform(189.09))
    #print(a)
    #a=np.reshape(a,(1,60,1))
    #Date = np.reshape(sc.transform(X_test[len(X_test)-1]),(1,60,1))
    #X_new='2019-04-23'
    #X_new1 = X_new[len(X_new) - 60]
    #X_new = np.reshape(X_new, (22,60, 1))
    #a=sc.transform(np.reshape(X_new,(1,60,1)))
    #print(regressor.predict(a))
