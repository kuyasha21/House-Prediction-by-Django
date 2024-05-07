from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Create your views here.

def home(req):
    return render(req, 'home.html')

def predict(req):
    return render(req, 'predict.html')

def result(req):
    
    d = pd.read_csv("D:\Code\Django\HousePricePrediction\House_prediction.csv")
    data = d.drop(['title','adress','type','purpose','flooPlan','url','lastUpdated'], axis=1)
    
    x = data.drop('price', axis=1)
    y = data['price']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30)
    
    model = LinearRegression()
    model.fit(x_train, y_train)

    beds = float(req.GET['beds'])
    baths = float(req.GET['baths'])
    area = float(req.GET['area'])

    pred = model.predict(np.array([beds, baths, area]).reshape(1,-1))
    pred = round(pred[0])

    # price = "2000"
    price = "The predicted price is " + str(pred) + " BDT"

    # predictions = model.predict(x_test)
    # predictions

    return render(req, 'predict.html', {"predictPrice" : price})

