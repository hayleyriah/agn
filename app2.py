import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

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

#-------------------------Multiple Linear Regression Model---------------------------------

# Encoding the target column using LabelEncoder
label_encoder = LabelEncoder()
df['Weekly_Sales_encoded'] = label_encoder.fit_transform(df['Weekly_Sales'])

# Use ColumnTransformer to encode the categorical features
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['Holiday_Flag'])], remainder='passthrough')  
X = df.drop(['Weekly_Sales', 'Weekly_Sales_encoded'], axis=1)
y = df['Weekly_Sales_encoded']
X_encoded = ct.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# Fit the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# User input for independent variables
st.sidebar.header("Enter values to be Predicted", divider='rainbow')

# Create the input for each feature
user_input = {}
for feature in X.columns:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}", 0.0)                        

# Button to trigger the prediction
if st.sidebar.button("Predict"):
    # Create a dataframe for the user input
    user_input_df = pd.DataFrame([user_input], dtype=float)
    
    # Transform the input using the fitted ColumnTransformer
    user_input_encoded = ct.transform(user_input_df)
    
    # Predict using the trained model
    y_pred = regressor.predict(user_input_encoded)
    
    # Inverse transform to get the original target values
    predicted_value = label_encoder.inverse_transform([int(y_pred)])
    
    # Display the predicted value
    st.header("Predicted Result Outcome:", divider='rainbow')
    st.write(predicted_value[0])