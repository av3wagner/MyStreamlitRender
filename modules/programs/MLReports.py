import streamlit as st  
import streamlit.components.v1 as components
stop=18
from datetime import datetime
import time
timestart = datetime.now()
stt = time.time()
now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
import scipy
import warnings

warnings.filterwarnings('ignore')
sns.set()

# preprocessing
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.calibration import CalibratedClassifierCV

# models
from sklearn.linear_model import LogisticRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report,f1_score,confusion_matrix,precision_score,recall_score,balanced_accuracy_score
import os, sys, inspect, time, datetime
import subprocess
import json
from time import time, strftime, localtime
from datetime import timedelta
import shutil

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

def plot_metrics(metrics_list):
    if metrics_list == 'Confusion Matrix':
        st.subheader('Confusion Matrix')
        ConfusionMatrixDisplay(model, X_test, y_test, display_labels=class_names)
        st.pyplot()

    if metrics_list == 'ROC Curve':
        st.subheader('ROC Curve')
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()

    if metrics_list == 'Precision-Recall Curve':
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()
        
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h1 style="color:yellow;font-size:28px;text-align:center">{"Автоматизировання Обучающая Система Научных Исследований в медицине и здравоохранении «АСНИ-Обучение»"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:white;font-size:24px;text-align:center">{"Тестовый пример проведения анализа данных c помощью Машинного Обучения (ML)"}</h2>', unsafe_allow_html=True)
            
today = datetime.date.today()
year = today.year

matrix ="/opt/render/project/src/data/heard.csv"
df = pd.read_csv(matrix) 

df.rename({'Y': 'target'}, axis=1, inplace=True)
df = df.fillna(0)

st.markdown("")
col1, col2, col3 = st.columns( [40, 1, 1])
with col1:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Набор данных для анализа"}</h2>', unsafe_allow_html=True)
    st.markdown("")

st.dataframe(df)  

train, test = train_test_split(df, test_size = 0.4)

X_train = train[train.columns.difference(['target'])]
y_train = train['target']

X_test = test[test.columns.difference(['target'])]
y_test = test['target']

cv_n_split = 2
random_state = 0
test_train_split_part = 0.2

train0, test0 = X_train, X_test
target0 = y_train
train, test, target, target_test = train_test_split(train0, target0, test_size=test_train_split_part, random_state=random_state)

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Базовая статистическа о данных"}</h2>', unsafe_allow_html=True)
st.markdown("")
st.write(test.describe())  

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
classifier = "Support Vector Machine (SVM)"
if classifier == 'Support Vector Machine (SVM)':
    C = 5.0            
    kernel = "sigmoid" 
    gamma = "auto"  
    clf = SVC(random_state=0)
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    st.write("Accuracy: ", accuracy)
    
    #st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names))
    #st.write("Precision: ", precision_score(y_test, y_pred)
    #st.write("Recall: ", recall_score(y_test, y_pred) #, labels=class_names))

    st.markdown("")
    col1, col2, col3 = st.columns( [40, 1, 1])
    with col1:  
        st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Confusion Matri для модели: Support Vector Machine (SVM)"}</h2>', unsafe_allow_html=True)
        st.markdown("")
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot()
               
    st.subheader('ROC Curve')
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.decision_function(X_test)
    RocCurveDisplay.from_predictions(y_test, y_pred)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot()    

    st.subheader('Precision-Recall Curve')
    PrecisionRecallDisplay.from_predictions(y_test, y_pred)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot() 

classifier = "Logistic Regression"
if classifier == "Logistic Regression":
    C = 5.0           
    max_iter = 200 
    model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    st.write("Accuracy: ", accuracy)
    st.markdown("")
    col1, col2, col3 = st.columns( [40, 1, 1])
    with col1:  
        st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Confusion Matri для модели: Logistic Regression"}</h2>', unsafe_allow_html=True)
        st.markdown("")
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot()
               
    st.subheader('ROC Curve')
    RocCurveDisplay.from_predictions(y_test, y_pred)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot()    

    #https://runebook.dev/en/docs/scikit_learn/modules/generated/sklearn.metrics.precisionrecalldisplay
    st.subheader('Precision-Recall Curve')
    PrecisionRecallDisplay.from_predictions(y_test, y_pred)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)      
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot() 

classifier = "Random Forest"
if classifier == 'Random Forest':
    n_estimators = 500
    max_depth = 10
    bootstrap = 'True' #'False'
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) #, bootstrap=bootstrap) #, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    st.write("Accuracy: ", accuracy)   

    st.markdown("")
    col1, col2, col3 = st.columns( [40, 1, 1])
    with col1:  
        st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Confusion Matri для модели: Random Forest"}</h2>', unsafe_allow_html=True)
        st.markdown("")
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot()
               
    st.subheader('ROC Curve')
    RocCurveDisplay.from_predictions(y_test, y_pred)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot()    
   
    st.subheader('Precision-Recall Curve')
    PrecisionRecallDisplay.from_predictions(y_test, y_pred)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)      
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot() 
        
    st.write('Ende Programm!')        
