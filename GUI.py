import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import pandas as pd
import os
import PIL


def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(df)
    os.startfile(csv_file_path)

def lstm():
    print("Running LSTM")
    root.title("LSTM in progress")
    #os.system('python apple_prediction_LSTM.py')
    from apple_prediction_LSTM import lstm_run
    lstm_run()
    
    root.title("LSTM Done")

def arima():
    print("Running Auto ARIMA")
    root.title("Auto ARIMA in progress")
    #os.system('python Apple_prediction_ARIMA.py')
    from Apple_prediction_ARIMA import arima_run
    arima_run()
    root.title("Auto ARIMA Done")

def sel():
   selection = "You selected the option " + str(var.get())
   label.config(text = selection)

def result():
    os.startfile("C:/Users/Nishant Sikri/Desktop/LSTM.png")
    os.startfile("C:/Users/Nishant Sikri/Desktop/ARIMA.png")
    os.startfile("C:/Users/Nishant Sikri/Desktop/comparison.csv")
    messagebox.showinfo("Result","According to the Analysis LSTM shows better accuracy")

root = tk.Tk()
root.geometry("500x500")
root.title("Stock Market Prediction")
tk.Label(root, text='File Path').grid(row=0, column=0)
v = tk.StringVar()
entry = tk.Entry(root, textvariable=v).grid(row=0, column=1)
tk.Button(root, text='Browse Data Set',command=import_csv_data,width=15,bg='SKYBLUE',fg='white',font=("Helvetica", 18)).place(x=10,y=50)
tk.Button(root, text='Close',command=root.destroy,width=15,bg='RED',fg='white',font=("Helvetica", 18)).place(x=260,y=50)

tk.Button(root, text='LSTM',command=lstm,width=15,bg='GREEN',fg='white',font=("Helvetica", 18)).place(x=10,y=150)
tk.Button(root, text='Auto ARIMA',command=arima,width=15,bg='pink',fg='white',font=("Helvetica", 18)).place(x=260,y=150)

tk.Button(root,text='Analysis Result',command=result,width=20,bg='GREEN',fg='white',font=("Helvetica", 18)).place(x=200,y=400)

root.mainloop()
