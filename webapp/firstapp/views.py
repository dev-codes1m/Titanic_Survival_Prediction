from django.shortcuts import render
import joblib
import os
import json
import pandas as pd
from .models import Titanic_Survival
import psycopg2


# Create your views here.

def index(request):
    return render(request,'index.html')

def result(request):
    model = joblib.load('../prediction_service/model/model.joblib')
    list = []
    list.append(float(request.GET['Pclass']))
    list.append(float(request.GET['Sex']))
    list.append(float(request.GET['Age']))
    list.append(float(request.GET['SibSp']))
    list.append(float(request.GET['Parch']))
    list.append(float(request.GET['Fare']))
    list.append(float(request.GET['Embarked']))
    list.append(float(request.GET['Fare_Tax']))
    list.append(float(request.GET['Food_Charges']))
    list.append(float(request.GET['Luggage_Charges']))
    answer = model.predict([list]).tolist()[0]

    b = Titanic_Survival(Pclass=request.GET['Pclass'],Sex=request.GET['Sex'],Age=request.GET['Age'],SibSp=request.GET['SibSp'],Parch=request.GET['Parch'],Fare=request.GET['Fare'],Embarked=request.GET['Embarked'],Fare_Tax=request.GET['Fare_Tax'],Food_Charges=request.GET['Food_Charges'],Luggage_Charges=request.GET['Luggage_Charges'],Survived=answer)
    b.save()

    return render(request,"index.html",{'answer':answer})
