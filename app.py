import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
st.title('Sales Forecasting Application')
st.header('original data')
data=pd.read_csv("E:\\data science course\\my_project\\retail_store_inventory.csv")
st.write(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data['month']=data['Date'].apply(lambda x:x.month)
data['day']=data['Date'].apply(lambda x:x.day)
data['Total Sales']=data['Units Sold']*data['Price']*(1-data['Discount']/100)
data = data.drop(columns=['Date','Demand Forecast'])
categorical_columns=['Category', 'Weather Condition', 'Seasonality','Region','Store ID','Product ID']
encode = LabelEncoder()
for column in categorical_columns:
    data[column]= encode.fit_transform(data[column])
st.header('Prepared data')
st.write(data.head())
st.sidebar.header('User Inputs')
st.header('Data description')
st.write(data.describe())
def user_inputs():
    Store_ID = st.sidebar.selectbox('Store_ID',[0,1,2,3,4])
    Product_ID = st.sidebar.slider('Product_ID',0,19,10)
    Category = st.sidebar.selectbox('Category',[0,1,2,3,4])
    Region = st.sidebar.selectbox('Region',[0,1,2,3])
    Inventory_Level = st.sidebar.slider('Inventory_Level',50, 500, 225)
    Units_Sold = st.sidebar.slider('Units_Sold',0, 250, 499)
    Units_Ordered = st.sidebar.slider('Units_Ordered',20,200,90)
    Price = st.sidebar.slider('Price',10.0,100.0,45.0)
    Discount = st.sidebar.slider('Discount',0,20,10)
    Weather_Condition= st.sidebar.selectbox('Weather_Condition',[0,1,2,3])
    Holiday_Promotion= st.sidebar.selectbox('Holiday_Promotion',[0,1])
    Competitor_Pricing= st.sidebar.slider('Competitor_Pricing',5.03,104.94,50.0)
    Seasonality = st.sidebar.selectbox('Seasonality',[0,1,2,3])
    month = st.sidebar.selectbox('month',[1,2,3,4,5,6,7,8,9,10,11,12])
    day = st.sidebar.slider('day',1, 31, 15)
    
    data = {
        'Store ID' : Store_ID,
        'Product ID' : Product_ID,
        'Category' :Category,
        'Region' : Region,
        'Inventory Level' : Inventory_Level,
        'Units Sold' : Units_Sold,
        'Units Ordered' : Units_Ordered,
        'Price' : Price,
        'Discount' : Discount,
        'Weather Condition' : Weather_Condition,
        'Holiday/Promotion' : Holiday_Promotion,
        'Competitor Pricing' : Competitor_Pricing,
        'Seasonality' : Seasonality,
        'month' : month,
        'day' : day
    }
    features = pd.DataFrame([data])
    return features
input_df = user_inputs()
st.header(
    """
    Predict Total Sales Based on Data Features
    """
)
x = data[['Store ID','Product ID','Category', 'Region', 'Inventory Level', 'Units Sold', 'Units Ordered','Price','Discount','Weather Condition','Holiday/Promotion',
          'Competitor Pricing','Seasonality','month','day']]
y = data['Total Sales']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 42)
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
st.write("R² Score:", r2_score(y_test, y_pred))

if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write('User Prediction: ', prediction)
