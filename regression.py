# Cross validation Regression MAE
# Evaluating using a Train and a Test sets
# Load CSV using Pandas
from pandas import read_csv # import pandas.read_csv as read
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def regression():
    filename = 'housing.csv'
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    array = dataframe.values

    # Separate array into input and output components
    X = array[:,0:13] # input
    Y = array[ :,13] # output
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    # Streamlit
    # Set up App

    st.subheader("ML Dashboard with Regression")
    st.write("Choose  Model from the options below:")

    classifier_list = ["LinearRegression",
                       "RidgeRegression",
                       "Lasso"]
    classifier = st.selectbox("Select classifier", classifier_list)
    # Add option to select classifiers
    # Add LogisticRegression
    if classifier == "LinearRegression":
        st.write("Here are the results of a linear regression model:")
        
        model = LinearRegression()
        scoringR2 = 'r2'
        resultsR2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoringR2)
        scoringMSE = 'neg_mean_squared_error'
        resultsMSE = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMSE)
        scoringMAE = 'neg_mean_absolute_error'
        resultsMAE = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMAE)
        # Display results
        metric_value = st.selectbox("Select metric",
                                    ["MAE",
                                     "MSE",
                                     "R^^2"])
        if metric_value == "MAE":
            st.write(f'MAE: {resultsMAE.mean()}')
        elif metric_value == "MSE":
            st.write(f'MSE: {resultsMSE.mean()}')
        elif metric_value == "R^^2":
            st.write(f'R^^2: {resultsR2.mean()}')
       

    
           
        
    elif classifier == "RidgeRegression":
        st.write("Here are the results of a Ridge regression model:")
        
        model = Ridge()
        scoringR2 = 'r2'
        resultsR2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoringR2)
        scoringMSE = 'neg_mean_squared_error'
        resultsMSE = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMSE)
        scoringMAE = 'neg_mean_absolute_error'
        resultsMAE = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMAE)
        # Display results
        metric_value = st.selectbox("Select metric",
                                    ["MAE",
                                     "MSE",
                                     "R^^2"])
        if metric_value == "MAE":
            st.write(f'MAE: {resultsMAE.mean()}')
        elif metric_value == "MSE":
            st.write(f'MSE: {resultsMSE.mean()}')
        elif metric_value == "R^^2":
            st.write(f'R^^2: {resultsR2.mean()}')
       

    
           

    elif classifier == "Lasso":
        st.write("Here are the results of a Lasso Regression model:")
        
        model = Lasso()
        scoringR2 = 'r2'
        resultsR2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoringR2)
        scoringMSE = 'neg_mean_squared_error'
        resultsMSE = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMSE)
        scoringMAE = 'neg_mean_absolute_error'
        resultsMAE = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMAE)
        # Display results
        metric_value = st.selectbox("Select metric",
                                    ["MAE",
                                     "MSE",
                                     "R^^2"])
        if metric_value == "MAE":
            st.write(f'MAE: {resultsMAE.mean()}')
        elif metric_value == "MSE":
            st.write(f'MSE: {resultsMSE.mean()}')
        elif metric_value == "R^^2":
            st.write(f'R^^2: {resultsR2.mean()}')
       

    
           