import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)


def classification():
    filename = 'pima-indians-diabetes.data.csv'
    attributs = ['preg', 'plas', 'pres', 'skin', 'test', 'mas', 'pedi', 'age', 'class']
    data = read_csv(filename, delimiter=",", names=attributs)
    #print(data)
    # Dataframe values
    array = data.values
    # Define inputs (X) and output (Y)
    X = array[ : , 0 : -1]
    Y = array[ : , -1]
    test_proportion = 0.30
    seed = 1000
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_proportion, random_state= seed)

    # Streamlit
    # Set up App

    st.subheader("ML Dashboard with Classification")
    st.write("Choose a transform type and Model from the options below:")

    classifier_list = ["LogisticRegression","DecisionTree"]
                       #"SVM",
                       #"KNeighbors",
                       #"RandomForest"
    classifier = st.selectbox("Select classifier", classifier_list)
    # Add option to select classifiers
    # Add LogisticRegression
    if classifier == "LogisticRegression":
        st.write("Here are the results of a logistic regression model:")
        solver_value = st.selectbox("Select solver",
                                    ["lbfgs",
                                     "liblinear",
                                     "newton-cg",
                                     "newton-cholesky"])
        model = LogisticRegression(solver=solver_value)
        model.fit(X_train, Y_train)
        # Make prediction
        
        y_pred = model.predict(X_test)
       
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, average='micro')
        recall = recall_score(Y_test, y_pred, average='macro')
        f1 = f1_score(Y_test, y_pred, average = "weighted")
        # Display results
        metric_value = st.selectbox("Select metric",
                                    ["accuracy",
                                     "precision",
                                     "recall",
                                     "f1"])
        if metric_value == "accuracy":
            st.write(f'Accuracy: {accuracy}')
        elif metric_value == "precision":
            st.write(f'Precision: {precision}')
        elif metric_value == "recall":
            st.write(f'Recall: {recall}')
        elif metric_value == "f1":
            st.write(f'F1-Score: {f1}')
        
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(Y_test, y_pred))


    elif classifier == "DecisionTree":
        st.write("Here are the results of a DecisionTree model:")
        
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        # Make prediction
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, average='micro')
        recall = recall_score(Y_test, y_pred, average='macro')
        f1 = f1_score(Y_test, y_pred, average = "weighted")
        # Display results
        metric_value = st.selectbox("Select metric",
                                    ["accuracy",
                                     "precision",
                                     "recall",
                                     "f1"])
        if metric_value == "accuracy":
            st.write(f'Accuracy: {accuracy}')
        elif metric_value == "precision":
            st.write(f'Precision: {precision}')
        elif metric_value == "recall":
            st.write(f'Recall: {recall}')
        elif metric_value == "f1":
            st.write(f'F1-Score: {f1}')

        
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(Y_test, y_pred))
           