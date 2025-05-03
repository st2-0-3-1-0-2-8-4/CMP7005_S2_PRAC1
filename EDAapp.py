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

# Set page configuration
st.set_page_config(
    page_title="Air Quality Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/st2-0-3-1-0-2-8-4/CMP7005_S2_PRAC1/refs/heads/main/combined_output.csv")
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
    

def eda():
    
    import pandas as pd
    st.title("Exploratory Data Analysis (EDA)")
    st.write("This page provides detailed exploratory data analysis.")

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum().to_frame().rename(columns={0: 'Missing Values'})
    missing_values['% of Total Values'] = 100 * missing_values['Missing Values'] / len(df)
    st.write(missing_values.style.background_gradient(cmap='Oranges'))
    
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
        
    selected_columns = ['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10', 'wd', 'DEWP', 'PRES', 'TEMP', 'RAIN', 'WSPM', 'year', 'day', 'month', 'hour', 'Date', 'station']
    
    # Create a new DataFrame with only the selected columns
    df1 = df[selected_columns]

    # Display the first few rows of the new DataFrame
    st.write(df1.head())
    pollutants = ['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10']
           
    df1['CO']=df1['CO'].fillna((df1['CO'].median()))
    df1['O3']=df1['O3'].fillna((df1['O3'].median()))
    df1['NO2']=df1['NO2'].fillna((df1['NO2'].median()))
    df1['SO2']=df1['SO2'].fillna((df1['SO2'].median()))
    df1['PM2.5']=df1['PM2.5'].fillna((df1['PM2.5'].median()))
    df1['PM10']=df1['PM10'].fillna((df1['PM10'].median()))
    df1['TEMP']=df1['TEMP'].fillna((df1['TEMP'].median()))
    df1['PRES']=df1['PRES'].fillna((df1['PRES'].median()))
    df1['DEWP']=df1['DEWP'].fillna((df1['DEWP'].median()))
    df1['RAIN']=df1['RAIN'].fillna((df1['RAIN'].median()))
    df1['wd']=df1['wd'].fillna((df1['wd'].mode()))
    df1['WSPM']=df1['WSPM'].fillna((df1['WSPM'].median()))

    
    # Create a new DataFrame with only the selected columns
    df1 = df[selected_columns]

    # Display the first few rows of the new DataFrame
    st.write(df1.head())
    
    stations = df['station'].value_counts()
    st.subheader(f'Total number of stations in the dataset : {len(stations)}')
    st.write(stations)

    df1.describe().T

    st.subheader("Key Insights from the Summary Statistics:")
    st.subheader("Date Range & Trends")
    
    st.write("The dataset spans from March 1, 2013, to Febraury 28, 2017. The median date (~December 31, 2014) suggests that most data points are centered around 2015-2017.")

    st.write("PM2.5 & PM10 Levels (Air Pollution Indicators)")

    st.write("PM2.5 Mean: 78.95 µg/m³ (with a max of 941.00 µg/m³ which indicates an extreme pollution event). Levels above 35 µg/m³ are considered unhealthy and precautions should be taken like wearing a face mask.")

    st.write("PM10 Mean: 104.82 µg/m³ (with a max of 999.00 that is considered hazardous for human health).")

    st.write("High standard deviation (79.69 for PM2.5 and 91.87 for PM10) suggest high air quality variations.")

    st.subheader("Other Pollutants")

    st.write("CO (Carbon Monoxide): Mean 1199.05 is considered potentially dangerous and can harm susceptible individuals (with a max of 10,000 which is considered hazardous).")

    st.write("O3 (Ozone): Mean 57.28 is considered moderate, max 450.00 that is considered very high and is dangerous to all and a health warning should be considered.")

    st.write("Nitric Oxide (NO2): Mean 50.00 is considered generally safe but prolonged exposure can affect vulnerable people, max 290.00 is considered concerning for human health.")

    st.write("Sulfur Dioxide (SO2): Mean 14.61is considered good air quality, max 500 represents a significant concern.")

    st.subheader("Final Thoughts:")

    st.write("The data suggests severe air pollution events, with occasional hazardous levels. High variability across pollutants implies that air quality is affected by multiple factors (seasonal changes, industrial activity, and vehicular emissions).")

    st.write("Further EDA with time-series analysis can help identify pollution trends and their causes.")

    st.subheader("Visualisation of each pollutant (using daily data)")
            
    # Visualizations
    st.subheader("Pollutant Levels Over Time")
    df1.set_index('Date', inplace=True)
    axes = df1[pollutants].plot(marker='.', alpha=0.5, linestyle='None', figsize=(16, 20), subplots=True)
    st.pyplot(plt)
    
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    pollutants = ['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10']
    
    # Monthly Pollutant Trends
    st.subheader("Monthly Pollutant Trends")
    df1['Month'] = df1.index.to_period('M')
    df_monthly = df1.groupby('Month')[pollutants].mean()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=["Pollutant Levels by Month"])
    for pollutant in pollutants:
        fig.add_trace(go.Scatter(x=df_monthly.index.astype(str), y=df_monthly[pollutant], mode='lines+markers', name=pollutant))
    fig.update_layout(title="Month-wise Pollutant Levels", xaxis_title="Month", yaxis_title="Pollutant Concentration (µg/m³)", height=600, showlegend=True)
    st.plotly_chart(fig)
    
    df1.reset_index(inplace=True)  # Moves 'Date' from index to a column

    # Ensure 'Date' column is datetime type
    df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')

    # Convert pollutant columns to numeric, coercing errors to NaN
    pollutant_columns = ['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10']
    for col in pollutant_columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')

    # Remove rows with NaN values (if needed)
    df1 = df1.dropna(subset=pollutant_columns)


    # Grouping by year and month, and calculating mean for each pollutant
    df1['Month'] = df1['Date'].dt.to_period('M')
    df_monthly = df1.groupby('Month')[pollutant_columns].mean()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    # Assuming df1 and pollutants are already defined
    # Group by Year and Month to calculate the monthly average for each pollutant
    monthly_avg = df1.groupby(['year', 'month'])[pollutants].mean().reset_index()

    # Create a Date column from Year and Month
    monthly_avg['Date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(DAY=1))

    # Create subplots for each pollutant
    fig = make_subplots(rows=len(pollutants), cols=1, subplot_titles=[f'{pollutant} Monthly Average Concentration Over Time' for pollutant in pollutants])

    # Add traces for each pollutant
    for i, pollutant in enumerate(pollutants):
        fig.add_trace(
            go.Scatter(
                x=monthly_avg['Date'],
                y=monthly_avg[pollutant],
                mode='lines+markers',
                name=pollutant,
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                opacity=0.7
            ),
            row=i+1, col=1
        )

    # Update layout
    fig.update_layout(
        title_text='Monthly Average Concentrations of Pollutants Over Time',
        title_font_size=24,
        showlegend=False,
        height=300 * len(pollutants),  # Adjust height based on the number of pollutants
    width=1000
    )

    # Update y-axis labels
    for i, pollutant in enumerate(pollutants):
        fig.update_yaxes(title_text=f'{pollutant} (ug/m3)', row=i+1, col=1)

    # Update x-axis labels
    fig.update_xaxes(title_text='Date', row=len(pollutants), col=1)

    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)

    # Show the plot
    st.plotly_chart(fig)
   
    # Dominant Pollutants
    st.subheader("Dominant Pollutants")
    pollutants_df = df1[pollutants].mean().to_frame().reset_index()
    pollutants_df.columns = ['Pollutant', 'Level']
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.pie(pollutants_df['Level'], labels=pollutants_df['Pollutant'], autopct='%1.1f%%', shadow=True, startangle=0)
    ax.axis('equal')
    st.pyplot(fig)
    
    # Top 2 Polluted Stations
    st.subheader("Top 2 Stations for Carbon Monoxide")
    top_stations = df1.groupby('station')['CO'].mean().sort_values(ascending=False).head(2).reset_index()
    st.write(top_stations)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_pollutants_df = df1[pollutants].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = numeric_pollutants_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)
    st.pyplot(plt)
    
    st.write("This shows some correlation between Carbon Monoxide and all other pollutants apart from Ozone")
    
    st.subheader("Correlation Heatmap of Weather")
    weather_and_pollutants = ['DEWP','PRES','TEMP', 'RAIN', 'WSPM', 'CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10']
    numeric_weather_df = df1[weather_and_pollutants]

    # Convert data to numeric (this will handle any non-numeric values)
    numeric_weather_df = numeric_weather_df.apply(pd.to_numeric, errors='coerce')

    # Calculate the correlation matrix
    correlation_matrix = numeric_weather_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Weather')
    st.pyplot(plt)
    
    st.write("This shows some correlation between Ozone and Temperature")

# Page 3: Model Building
def model_building(df):
    st.title("Model Building")
    st.write("This page predicts CO for the next day based on user input.")

    # Handle missing values
    pollutants = ['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10']
    for col in pollutants:
        df[col] = df[col].fillna(df[col].median())  # Impute missing values with median
    
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    selected_columns = ['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10', 'Date']
    
    # Prepare data for modeling
    df_model = df[['CO', 'O3', 'NO2', 'SO2', 'PM2.5', 'PM10', 'Date']].dropna()
        
    # Feature engineering
    df_model['Day'] = df_model['Date'].dt.day
    df_model['Month'] = df_model['Date'].dt.month
    df_model['Year'] = df_model['Date'].dt.year
    df_model = df_model.drop('Date', axis=1)

    # Split data
    X = df_model.drop('CO', axis=1)
    y = df_model['CO']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    # Observed vs. Predicted Table
    results_df = pd.DataFrame({"Observed CO": y_test, "Predicted CO": y_pred})
    results_df.reset_index(drop=True, inplace=True)
    st.subheader("Observed vs. Predicted CO Values")
    st.write(results_df.head(20))  # Display first 20 values

    # Line Plot for Observed vs. Predicted
    fig = px.line(results_df, title="Observed vs. Predicted CO",
                  labels={"index": "Sample Index", "value": "CO"},
                  markers=True)
    fig.add_scatter(x=results_df.index, y=results_df["Observed CO"], mode='lines+markers', name="Observed")
    fig.add_scatter(x=results_df.index, y=results_df["Predicted CO"], mode='lines+markers', name="Predicted")
    st.plotly_chart(fig)

    # User input for predictions
    st.subheader("Enter values for next-day prediction")
    o3 = st.number_input("O3", min_value=0.0, max_value=500.0, value=float(df_model['O3'].median()))
    no2 = st.number_input("NO2", min_value=0.0, max_value=500.0, value=float(df_model['NO2'].median()))
    so2 = st.number_input("SO2", min_value=0.0, max_value=1000.0, value=float(df_model['SO2'].median()))
    pm2 = st.number_input("PM2.5", min_value=0.0, max_value=1000.0, value=float(df_model['PM2.5'].median()))
    pm10 = st.number_input("PM10", min_value=0.0, max_value=1000.0, value=float(df_model['PM10'].median()))

    if st.button("Predict CO for the Next Day"):
        # Create a DataFrame for the next day
        next_day = pd.DataFrame({
            'Day': [df['Date'].max().day + 1],  # Next day
            'Month': [df['Date'].max().month],
            'Year': [df['Date'].max().year],
            'O3': [o3],
            'NO2': [no2],
            'SO2': [so2],
            'PM2.5': [pm2],
            'PM10': [pm10]
        })

        # Ensure the feature names match the training data
        next_day = next_day[X_train.columns]

        # Predict CO for the next day
        predicted_co = model.predict(next_day)

        # Display prediction
        st.subheader("CO Prediction for the Next Day")
        st.write(f"Predicted CO: {predicted_co[0]:.2f} µg/m³")

        # Optional: Display the input values for reference
        st.write("Input Values Used for Prediction:")
        st.write(next_day)

# Main App
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.write("Select a page to explore the air quality dataset.")

    # Page selection
    page = st.sidebar.radio(
        "Go to",
        ["Home", "General Data Information", "Exploratory Data Analysis (EDA)", "Model Building"],
        index=0  # Default to Home page
    )

    # Home Page
    if page == "Home":
        st.title("Air Quality Analysis App")
        st.write("Welcome to the Air Quality Analysis App! This app allows you to explore air quality data and predict CO levels for the next day.")
        st.write("### Features:")
        st.write("- **General Data Information**: Overview of the dataset.")
        st.write("- **Exploratory Data Analysis (EDA)**: Visualize trends, correlations, and dominant pollutants.")
        st.write("- **Model Building**: Predict CO levels for the next day based on user inputs.")
        st.write("### How to Use:")
        st.write("1. Use the sidebar to navigate between pages.")
        st.write("2. Explore the dataset and insights in the **General Data Information** and **EDA** pages.")
        st.write("3. Go to the **Model Building** page to predict CO for the next day.")
        st.write("---")
        

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
