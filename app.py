# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
# Import classification
from classification import classification
from regression import regression

# Streamlit
# Set up App
st.set_page_config(page_title=" ML Dashboard", 
                   layout="centered",
                   initial_sidebar_state="auto")
# Add title and markdown decription
st.title("Predictive Modelling Dashboard : Classification & Regression")

# define sidebar and sidebar options
options = ["Classification", "Regression"]
selected_option = st.sidebar.selectbox("Select an option", options)
# Do Classification
if selected_option == "Classification":
    # Call/invoke classification function from classification.py
    classification()
# Predictive Modelling
elif selected_option == "Regression":
     regression()