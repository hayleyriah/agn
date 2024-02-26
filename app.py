import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("Walmart.csv")

# Add logo
st.image("Image.jpg")

# Add title to app
st.title("Walmart Prediction App")

#Add the header

st.header("Dataset Concept.", divider="rainbow")

#Add paragraph explaining the dataset

st.write("""
The dataset encompasses various features related to 
retail sales and economic indicators. These features offer insights into 
sales performance relative to economic and environmental factors, facilitating retail analysis and 
forecasting""")

#------------------------------------------------------DISPLAY EDA----------------------------------------------

st.header("Exploratory Data Analysis (EDA)", divider="rainbow")


if st.checkbox("Dataset info"):
     st.write("Dataset info", df.info())
     
if st.checkbox("Number of Rows"):
     st.write("Number of Rows", df.shape[0])
     
if st.checkbox("Number of Columns"):
     st.write("Number of Columns", df.columns.tolist())
     
if st.checkbox("Data types"):
     st.write("Data types", df.dtypes)
     
if st.checkbox("Missing Values"):
     st.write("Missing Values", df.isnull().sum())
     
if st.checkbox("Statistical Summary"):
     st.write("Statistical Summary", df.describe())

#==============================================Visualization===============================

st.header("VIsualization of the Dataset (VIZ)", divider="rainbow")

# Checkbox to trigger bar chart
if st.checkbox("Generate Bar Chart"):
    selected_columns = st.multiselect("Select the columns to visualize the Bar Chart", df.columns)
    if selected_columns:
        st.bar_chart(df[selected_columns])
    else:
        st.warning("Please select at least one column.")

# Checkbox for line chart
if st.checkbox("Generate Line Chart"):
    selected_columns = st.multiselect("Select the columns to visualize the Line Chart", df.columns)
    if selected_columns:
        st.line_chart(df[selected_columns])
    else:
        st.warning("Please select at least one column.")
        

# Prepare the dataset
X=df.drop("Holiday_Flag", axis = 1)
y = df["Holiday_Flag"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


#create and fit model

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)


