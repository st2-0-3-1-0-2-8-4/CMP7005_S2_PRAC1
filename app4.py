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

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['Year'] = pd.DatetimeIndex(df['Date']).year
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

# Page 2: EDA
def eda():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("This page provides detailed exploratory data analysis.")

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum().to_frame().rename(columns={0: 'Missing Values'})
    missing_values['% of Total Values'] = 100 * missing_values['Missing Values'] / len(df)
    st.write(missing_values.style.background_gradient(cmap='Oranges'))

    # Create new columns
    df['PM'] = df['PM10'] + df['PM2.5']
    df['Nitric'] = df['NO'] + df['NO2'] + df['NOx']
    df['BTX'] = df['Benzene'] + df['Toluene'] + df['Xylene']

    # New DataFrame for AQI
    selected_columns = ['City', 'Date', 'month', 'Year', 'PM2.5', 'PM10', 'Nitric', 'NH3', 'CO', 'SO2', 'O3', 'BTX', 'AQI', 'AQI_Bucket']
    df1 = df[selected_columns]

    # Impute missing values
    pollutants = ['PM2.5', 'PM10', 'Nitric', 'NH3', 'CO', 'SO2', 'O3', 'BTX', 'AQI']
    for col in pollutants:
        df1[col] = df1[col].fillna(df1[col].median())
    df1['AQI_Bucket'] = df1['AQI_Bucket'].fillna('moderate')

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df1.describe().T)

    # Visualizations
    st.subheader("Pollutant Levels Over Time")
    df1.set_index('Date', inplace=True)
    pollutants = ['PM2.5', 'PM10', 'Nitric', 'NH3', 'CO', 'SO2', 'O3', 'BTX']
    axes = df1[pollutants].plot(marker='.', alpha=0.5, linestyle='None', figsize=(16, 20), subplots=True)
    st.pyplot(plt)

    # Monthly AQI Trends
    st.subheader("Monthly AQI Trends")
    df1['Month'] = df1.index.to_period('M')
    df_monthly = df1.groupby('Month')[pollutants].mean()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=["Pollutant Levels by Month"])
    for pollutant in pollutants:
        fig.add_trace(go.Scatter(x=df_monthly.index.astype(str), y=df_monthly[pollutant], mode='lines+markers', name=pollutant))
    fig.update_layout(title="Month-wise Pollutant Levels", xaxis_title="Month", yaxis_title="Pollutant Concentration (µg/m³)", height=600, showlegend=True)
    st.plotly_chart(fig)

    # Dominant Pollutants
    st.subheader("Dominant Pollutants")
    pollutants_df = df1[pollutants].mean().to_frame().reset_index()
    pollutants_df.columns = ['Pollutant', 'Level']
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.pie(pollutants_df['Level'], labels=pollutants_df['Pollutant'], autopct='%1.1f%%', shadow=True, startangle=0)
    ax.axis('equal')
    st.pyplot(fig)

    # Top 10 Polluted Cities
    st.subheader("Top 10 Polluted Cities")
    top_cities = df1.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10).reset_index()
    st.write(top_cities)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_pollutants_df = df1[pollutants].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = numeric_pollutants_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)
    st.pyplot(plt)

# Page 3: Model Building
def model_building(df):
    st.title("Model Building")
    st.write("This page predicts PM2.5 for the next day based on user input.")

    # Ensure 'Nitric' and 'BTX' columns exist
    if 'Nitric' not in df.columns:
        df['Nitric'] = df['NO'] + df['NO2'] + df['NOx']
    if 'BTX' not in df.columns:
        df['BTX'] = df['Benzene'] + df['Toluene'] + df['Xylene']

    # Handle missing values
    pollutants = ['PM2.5', 'PM10', 'Nitric', 'NH3', 'CO', 'SO2', 'O3', 'BTX', 'AQI']
    for col in pollutants:
        df[col] = df[col].fillna(df[col].median())  # Impute missing values with median
    df['AQI_Bucket'] = df['AQI_Bucket'].fillna('moderate')  # Fill missing AQI_Bucket

    # Prepare data for modeling
    df_model = df[['Date', 'PM2.5', 'PM10', 'Nitric', 'NH3', 'CO', 'SO2', 'O3', 'BTX']].dropna()
    df_model['Date'] = pd.to_datetime(df_model['Date'])

    # Feature engineering
    df_model['Day'] = df_model['Date'].dt.day
    df_model['Month'] = df_model['Date'].dt.month
    df_model['Year'] = df_model['Date'].dt.year
    df_model = df_model.drop('Date', axis=1)

    # Split data
    X = df_model.drop('PM2.5', axis=1)
    y = df_model['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    # Observed vs. Predicted Table
    results_df = pd.DataFrame({"Observed PM2.5": y_test, "Predicted PM2.5": y_pred})
    results_df.reset_index(drop=True, inplace=True)
    st.subheader("Observed vs. Predicted PM2.5 Values")
    st.write(results_df.head(20))  # Display first 20 values

    # Line Plot for Observed vs. Predicted
    fig = px.line(results_df, title="Observed vs. Predicted PM2.5",
                  labels={"index": "Sample Index", "value": "PM2.5"},
                  markers=True)
    fig.add_scatter(x=results_df.index, y=results_df["Observed PM2.5"], mode='lines+markers', name="Observed")
    fig.add_scatter(x=results_df.index, y=results_df["Predicted PM2.5"], mode='lines+markers', name="Predicted")
    st.plotly_chart(fig)

    # User input for predictions
    st.subheader("Enter values for next-day prediction")
    pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=float(df_model['PM10'].median()))
    nitric = st.number_input("Nitric (NO + NO2 + NOx)", min_value=0.0, max_value=500.0, value=float(df_model['Nitric'].median()))
    nh3 = st.number_input("NH3", min_value=0.0, max_value=500.0, value=float(df_model['NH3'].median()))
    co = st.number_input("CO", min_value=0.0, max_value=500.0, value=float(df_model['CO'].median()))
    so2 = st.number_input("SO2", min_value=0.0, max_value=500.0, value=float(df_model['SO2'].median()))
    o3 = st.number_input("O3", min_value=0.0, max_value=500.0, value=float(df_model['O3'].median()))
    btx = st.number_input("BTX (Benzene + Toluene + Xylene)", min_value=0.0, max_value=500.0, value=float(df_model['BTX'].median()))

    if st.button("Predict PM2.5 for the Next Day"):
        # Create a DataFrame for the next day
        next_day = pd.DataFrame({
            'Day': [df['Date'].max().day + 1],  # Next day
            'Month': [df['Date'].max().month],
            'Year': [df['Date'].max().year],
            'PM10': [pm10],
            'Nitric': [nitric],
            'NH3': [nh3],
            'CO': [co],
            'SO2': [so2],
            'O3': [o3],
            'BTX': [btx]
        })

        # Ensure the feature names match the training data
        next_day = next_day[X_train.columns]

        # Predict PM2.5 for the next day
        predicted_pm25 = model.predict(next_day)

        # Display prediction
        st.subheader("PM2.5 Prediction for the Next Day")
        st.write(f"Predicted PM2.5: {predicted_pm25[0]:.2f} µg/m³")

        # Optional: Display the input values for reference
        st.write("Input Values Used for Prediction:")
        st.write(next_day)

# Main App
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.write("Select a page to explore the air quality dataset.")

    # Add a logo or image to the sidebar (optional)
    st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Replace with your logo URL

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