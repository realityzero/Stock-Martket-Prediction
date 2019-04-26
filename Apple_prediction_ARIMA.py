def arima_run():

    #import packages
    import pandas as pd
    import numpy as np
    import pickle
    import csv

    #to plot within notebook
    import matplotlib.pyplot as plt

    #setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    #for normalizing data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    #read the file
    df = pd.read_csv('arimaAPPLE.csv')

    #print the head
    df.head()

    #setting index as date
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']

    #plot
    plt.figure(figsize=(16,8))
    #plt.plot(df['Close'], label='Close Price history')

    from pyramid.arima import auto_arima

    data = df.sort_index(ascending=True, axis=0)

    train = data[:9634]
    valid = data[9634:]

    training = train['Open']
    validation = valid['Open']

    #model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
    #model.fit(training)


    # Serialize with Pickle
    #with open('arima.pkl', 'wb') as pkl:
    #    pickle.dump(model, pkl)



    # Now read it back and make a prediction
    with open('arima.pkl', 'rb') as pkl:
        stepwise_model = pickle.load(pkl)

    print("Model Loaded")
    
    forecast = stepwise_model.predict(n_periods=21)
    forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

    #rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))
    #print(rms)
    print(forecast['Prediction'].tolist())

    with open('comparison.csv', newline='') as f:
        reader = csv.reader(f)
        data = [line for line in reader]
        #for val in dataset_test['Open']:
        #    writer.writerow([val])

    ls = forecast['Prediction'].tolist()
    list3 = [list(a) for a in zip(data, ls)]

    print(list3)
    
    with open('comparison.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['LSTM','ARIMA'])
        w.writerows(list3)
        #for val in ls:
        #    w.writerow([val])
        #w.writerows(forecast['Prediction'].tolist())

    #plot
    plt.plot(valid['Open'], color = 'black', label = 'APPLE Stock Price')
    plt.plot(forecast['Prediction'], color = 'green', label = 'Predicted APPLE Stock Price')
    plt.xlabel('Date')
    plt.ylabel('APPLE Stock Price')
    plt.legend()
    plt.show()
