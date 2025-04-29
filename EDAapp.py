import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set page configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Air Quality Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv("combined_output.csv")
    return df

df = load_data()

# Page 1: General Data Information
def general_info():
    st.title("General Data Information")
    st.write("This page provides an overview of the dataset.")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

# Main App
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.write("Select a page to explore the air quality dataset.")

    # Add a logo or image to the sidebar (optional)
    # st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Replace with your logo URL

    # Page selection
    page = st.sidebar.radio(
        "Go to",
        ["Home", "General Data Information", "Exploratory Data Analysis (EDA)", "Model Building"],
        index=0  # Default to Home page
    )

    # Home Page
    if page == "Home":
        st.title("🌍 Air Quality Analysis App")
        st.write("Welcome to the Air Quality Analysis App! This app allows you to explore air quality data and predict PM2.5 levels for the next day.")
        st.write("### Features:")
        st.write("- **General Data Information**: Overview of the dataset.")
        st.write("- **Exploratory Data Analysis (EDA)**: Visualize trends, correlations, and dominant pollutants.")
        st.write("- **Model Building**: Predict PM2.5 levels for the next day based on user inputs.")
        st.write("### How to Use:")
        st.write("1. Use the sidebar to navigate between pages.")
        st.write("2. Explore the dataset and insights in the **General Data Information** and **EDA** pages.")
        st.write("3. Go to the **Model Building** page to predict PM2.5 for the next day.")
        st.write("---")
        st.write("Made with ❤️ using Streamlit")

    # General Data Information Page
    elif page == "General Data Information":
        general_info()

    # Exploratory Data Analysis (EDA) Page
    elif page == "Exploratory Data Analysis (EDA)":
        eda()

    # Model Building Page
    elif page == "Model Building":
        model_building(df)

if __name__ == "__main__":
    main()