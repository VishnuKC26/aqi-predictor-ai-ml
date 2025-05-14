import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Indian AQI Prediction App", page_icon="🌬️", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Make Prediction", "AQI Forecast"])

# Mode Selection
st.sidebar.subheader("Prediction Mode")
prediction_mode = st.sidebar.radio("Select Mode", ["Normal (Weather + PM2.5)", "Scientific (Weather + Pollutants)"], index=0)
is_scientific_mode = prediction_mode == "Scientific (Weather + Pollutants)"

# City Selection
st.sidebar.subheader("City Selection")
try:
    df_temp = pd.read_csv("city_day.csv")
    available_cities = sorted(df_temp['City'].unique().tolist())
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure 'city_day.csv' is in the directory.")
    st.markdown("Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).")
    available_cities = ['Delhi', 'Ahmedabad']

selected_city = st.sidebar.selectbox("Select City", available_cities, index=available_cities.index('Delhi') if 'Delhi' in available_cities else 0)

# Title and description
st.title(f"{selected_city} Air Quality Index (AQI) Prediction System")
st.markdown(f"""
Predicts AQI for {selected_city} using {'weather and predicted PM2.5' if not is_scientific_mode else 'weather and pollutant parameters'}.
Mode: {'Normal (Weather + PM2.5)' if not is_scientific_mode else 'Scientific (Weather + Pollutants)'}.
Select a model from the sidebar to train and make predictions.
""")

# Load the dataset
@st.cache_data
def load_data(selected_city):
    data_path = "city_day.csv"
    
    try:
        df = pd.read_csv(data_path)
        st.write(f"Dataset loaded successfully. Available cities: {df['City'].unique().tolist()}")
        if selected_city not in df['City'].values:
            st.warning(f"No data found for {selected_city}. Using synthetic data.")
            raise ValueError("City not found")
        df = df[df['City'] == selected_city]
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Error loading dataset for {selected_city}: {e}")
        st.markdown("Please ensure 'city_day.csv' is in the directory. Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).")
        df = pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d')],
            'City': [selected_city],
            'AQI': [300],
            'PM2.5': [120],
            'PM10': [200],
            'NO': [50],
            'NO2': [60],
            'NOx': [110],
            'CO': [1.5],
            'SO2': [20],
            'O3': [30],
            'Benzene': [5],
            'Toluene': [10],
            'AQI_Bucket': ['Very Poor']
        })
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    
    # Synthetic weather data
    temp_means = {1: 15, 2: 18, 3: 23, 4: 30, 5: 36, 6: 38, 
                  7: 34, 8: 33, 9: 32, 10: 28, 11: 22, 12: 17}
    temp_vars = {1: 5, 2: 5, 3: 6, 4: 6, 5: 5, 6: 4, 
                 7: 3, 8: 3, 9: 4, 10: 5, 11: 5, 12: 5}
    humidity_means = {1: 75, 2: 65, 3: 55, 4: 40, 5: 35, 6: 50, 
                     7: 75, 8: 80, 9: 70, 10: 60, 11: 65, 12: 70}
    wind_means = {1: 7, 2: 8, 3: 9, 4: 10, 5: 10, 6: 12, 
                 7:  9, 8: 8, 9: 7, 10: 6, 11: 5, 12: 6}
    pressure_means = {1: 1018, 2: 1016, 3: 1013, 4: 1010, 5: 1006, 6: 1000, 
                     7: 996, 8: 998, 9: 1002, 10: 1010, 11: 1014, 12: 1017}
    
    if selected_city == 'Ahmedabad':
        temp_means = {1: 20, 2: 23, 3: 28, 4: 33, 5: 37, 6: 36,
                      7: 33, 8: 32, 9: 32, 10: 30, 11: 26, 12: 22}
        humidity_means = {1: 60, 2: 55, 3: 50, 4: 40, 5: 35, 6: 45,
                         7: 70, 8: 75, 9: 65, 10: 55, 11: 50, 12: 55}
    
    np.random.seed(42)
    df['Temperature'] = df['month'].apply(lambda m: np.random.normal(temp_means[m], temp_vars[m]))
    df['Humidity'] = df['month'].apply(lambda m: min(100, max(30, np.random.normal(humidity_means[m], 10))))
    df['Wind_Speed'] = df['month'].apply(lambda m: max(0, np.random.normal(wind_means[m], 3)))
    df['Pressure'] = df['month'].apply(lambda m: np.random.normal(pressure_means[m], 3))
    
    # Time-based features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['indian_season'] = df['month'].map({
        12: 1, 1: 1, 2: 1,  # Winter
        3: 2, 4: 2, 5: 2,   # Summer
        6: 3, 7: 3, 8: 3, 9: 3,  # Monsoon
        10: 4, 11: 4  # Post-Monsoon
    })
    
    # Fill missing values for pollutants and AQI
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    for col in pollutants:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if df['AQI'].isna().sum() > 0 and 'PM2.5' in df.columns:
        df['AQI'] = df['AQI'].fillna(df['PM2.5'] * 3)
    
    df = df.dropna(subset=['AQI', 'Temperature', 'Humidity', 'PM2.5'] + pollutants)
    return df

# Train PM2.5 prediction model (New)
@st.cache_resource
def train_pm25_model(data):
    X, _, _ = prepare_features(data, is_scientific_mode=False, include_pm25=False)
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'pm25_model.pkl')
    mse = mean_squared_error(y_test, model.predict(X_test))
    st.write(f"PM2.5 Prediction Model MSE: {mse:.2f}")
    return model

# Predict PM2.5 (New)
def predict_pm25(model, input_data):
    X_input, _, _ = prepare_features(input_data, is_scientific_mode=False, include_pm25=False)
    X_train, _, _ = prepare_features(data, is_scientific_mode=False, include_pm25=False)
    input_features = pd.DataFrame(columns=X_train.columns)
    for col in input_features.columns:
        input_features[col] = input_data[col].values if col in input_data.columns else 0
    return max(0, model.predict(input_features)[0])

# Prepare features (Modified to include PM2.5 in normal mode)
def prepare_features(df, is_scientific_mode=False, include_pm25=True):
    features = pd.DataFrame()
    time_features = ['month', 'day_of_week', 'is_weekend', 'indian_season']
    weather_features = ['Temperature', 'Humidity', 'Wind_Speed', 'Pressure']
    pollutant_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'] if is_scientific_mode else ['PM2.5'] if include_pm25 else []
    
    selected_features = time_features + weather_features + pollutant_features
    
    for feature in selected_features:
        if feature in df.columns:
            features[feature] = df[feature]
    
    return features, weather_features, pollutant_features

# Apply feature weights for scientific mode
def apply_feature_weights(X, weather_features, weight=2.0):
    X_weighted = X.copy()
    for feature in weather_features:
        if feature in X.columns:
            X_weighted[feature] = X_weighted[feature] * weight
    return X_weighted

# AQI interpretation functions
def get_health_implications_india(category):
    implications = {
        "Good": "Air quality is satisfactory, minimal risk.",
        "Satisfactory": "Minor breathing discomfort to sensitive people.",
        "Moderate": "Breathing discomfort to people with lung disease.",
        "Poor": "Breathing discomfort to most on prolonged exposure.",
        "Very Poor": "Respiratory illness on prolonged exposure.",
        "Severe": "Serious health effects, avoid outdoor activity."
    }
    return implications.get(category, "No information available")

def get_cautionary_statement_india(category):
    statements = {
        "Good": "Enjoy outdoor activities.",
        "Satisfactory": "Sensitive groups reduce heavy exertion.",
        "Moderate": "Limit prolonged outdoor exertion for sensitive groups.",
        "Poor": "Limit exertion, avoid symptoms like coughing.",
        "Very Poor": "Avoid all outdoor exertion, close windows.",
        "Severe": "Emergency conditions, stay indoors."
    }
    return statements.get(category, "No information available")

# Download model
def download_model(model, model_name, is_scientific_mode):
    mode_suffix = "_scientific" if is_scientific_mode else "_normal"
    model_file = f"{model_name.lower().replace(' ', '_')}_aqi_model{mode_suffix}.pkl"
    joblib.dump(model, model_file)
    with open(model_file, "rb") as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name=model_file,
            mime="application/octet-stream"
        )

# Generate forecast (Modified to include predicted PM2.5)
def generate_forecast(model, normal_model, pm25_model, days=7, city='Delhi', is_scientific_mode=False):
    today = datetime.now().date()
    forecast_dates = [today + timedelta(days=i) for i in range(days)]
    
    forecast_data = []
    
    temp_means = {1: 15, 2: 18, 3: 23, 4: 30, 5: 36, 6: 38, 
                  7: 34, 8: 33, 9: 32, 10: 28, 11: 22, 12: 17}
    humidity_means = {1: 75, 2: 65, 3: 55, 4: 40, 5: 35, 6: 50, 
                     7: 75, 8: 80, 9: 70, 10: 60, 11: 65, 12: 70}
    wind_means = {1: 7, 2: 8, 3: 9, 4: 10, 5: 10, 6: 12, 
                 7: 9, 8: 8, 9: 7, 10: 6, 11: 5, 12: 6}
    pressure_means = {1: 1018, 2: 1016, 3: 1013, 4: 1010, 5: 1006, 6: 1000, 
                     7: 996, 8: 998, 9: 1002, 10: 1010, 11: 1014, 12: 1017}
    pm25_means = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 6: 50, 
                  7: 60, 8: 70, 9: 80, 10: 90, 11: 100, 12: 110}
    pm10_means = {1: 150, 2: 140, 3: 130, 4: 120, 5: 110, 6: 100, 
                  7: 110, 8: 120, 9: 130, 10: 140, 11: 150, 12: 160}
    no2_means = {1: 50, 2: 48, 3: 45, 4: 42, 5: 40, 6: 38, 
                 7: 40, 8: 42, 9: 45, 10: 48, 11: 50, 12: 52}
    so2_means = {1: 20, 2: 19, 3: 18, 4: 17, 5: 16, 6: 15, 
                 7: 16, 8: 17, 9: 18, 10: 19, 11: 20, 12: 21}
    co_means = {1: 1.5, 2: 1.4, 3: 1.3, 4: 1.2, 5: 1.1, 6: 1.0, 
                7: 1.1, 8: 1.2, 9: 1.3, 10: 1.4, 11: 1.5, 12: 1.6}
    o3_means = {1: 30, 2: 32, 3: 35, 4: 38, 5: 40, 6: 42, 
                7: 40, 8: 38, 9: 35, 10: 32, 11: 30, 12: 28}
    
    if city == 'Ahmedabad':
        temp_means = {1: 20, 2: 23, 3: 28, 4: 33, 5: 37, 6: 36,
                      7: 33, 8: 32, 9: 32, 10: 30, 11: 26, 12: 22}
        humidity_means = {1: 60, 2: 55, 3: 50, 4: 40, 5: 35, 6: 45,
                         7: 70, 8: 75, 9: 65, 10: 55, 11: 50, 12: 55}
    
    for date in forecast_dates:
        month = date.month
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        indian_season_map = {
            12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4
        }
        indian_season = indian_season_map.get(month, 1)
        
        input_data = pd.DataFrame({
            'month': [month],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'indian_season': [indian_season],
            'Temperature': [np.random.normal(temp_means[month], 2)],
            'Humidity': [min(100, max(30, np.random.normal(humidity_means[month], 5)))],
            'Wind_Speed': [max(0, np.random.normal(wind_means[month], 1))],
            'Pressure': [np.random.normal(pressure_means[month], 1)],
            'PM2.5': [predict_pm25(pm25_model, pd.DataFrame({
                'month': [month],
                'day_of_week': [day_of_week],
                'is_weekend': [is_weekend],
                'indian_season': [indian_season],
                'Temperature': [np.random.normal(temp_means[month], 2)],
                'Humidity': [min(100, max(30, np.random.normal(humidity_means[month], 5)))],
                'Wind_Speed': [max(0, np.random.normal(wind_means[month], 1))],
                'Pressure': [np.random.normal(pressure_means[month], 1)]
            }))],
            'PM10': [max(0, np.random.normal(pm10_means[month], 15))] if is_scientific_mode else [0],
            'NO2': [max(0, np.random.normal(no2_means[month], 5))] if is_scientific_mode else [0],
            'SO2': [max(0, np.random.normal(so2_means[month], 3))] if is_scientific_mode else [0],
            'CO': [max(0, np.random.normal(co_means[month], 0.2))] if is_scientific_mode else [0],
            'O3': [max(0, np.random.normal(o3_means[month], 5))] if is_scientific_mode else [0]
        })
        
        X_input, weather_features, _ = prepare_features(input_data, is_scientific_mode)
        
        input_features = pd.DataFrame(columns=X_input.columns)
        for col in input_features.columns:
            input_features[col] = input_data[col].values if col in input_data.columns else 0
        
        if is_scientific_mode:
            input_features = apply_feature_weights(input_features, weather_features, weight=2.0)
            prediction = model.predict(input_features)[0]
            X_normal, _, _ = prepare_features(input_data, is_scientific_mode=False)
            normal_features = pd.DataFrame(columns=X_normal.columns)
            for col in normal_features.columns:
                normal_features[col] = input_data[col].values if col in input_data.columns else 0
            normal_pred = normal_model.predict(normal_features)[0]
            deviation_penalty = 0.1 * (prediction - normal_pred)**2
            prediction = prediction - deviation_penalty / (1 + abs(prediction))
        else:
            prediction = model.predict(input_features)[0]
        
        category = "Good" if prediction <= 50 else "Satisfactory" if prediction <= 100 else "Moderate" if prediction <= 200 else "Poor" if prediction <= 300 else "Very Poor" if prediction <= 400 else "Severe"
        
        forecast_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Day': date.strftime('%A'),
            'Predicted AQI': round(prediction, 2),
            'Category': category,
            'Temperature': round(input_data['Temperature'][0], 1),
            'Humidity': round(input_data['Humidity'][0], 1),
            'Wind Speed': round(input_data['Wind_Speed'][0], 1),
            'PM2.5': round(input_data['PM2.5'][0], 1),
            'PM10': round(input_data['PM10'][0], 1) if is_scientific_mode else None,
            'NO2': round(input_data['NO2'][0], 1) if is_scientific_mode else None,
            'SO2': round(input_data['SO2'][0], 1) if is_scientific_mode else None,
            'CO': round(input_data['CO'][0], 2) if is_scientific_mode else None,
            'O3': round(input_data['O3'][0], 1) if is_scientific_mode else None
        })
    
    return pd.DataFrame(forecast_data)

# Export forecast to CSV
def export_to_csv(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name="aqi_forecast.csv",
        mime="text/csv",
    )

# Load data
try:
    data = load_data(selected_city)
except Exception as e:
    st.error(f"Error loading data for {selected_city}: {e}")
    data = pd.DataFrame()

# Train PM2.5 model
if not data.empty:
    pm25_model = train_pm25_model(data)
else:
    pm25_model = None

# Data Exploration Page
if page == "Data Exploration":
    st.header(f"Data Exploration for {selected_city}")
    
    if not data.empty:
        st.subheader("Sample Data")
        st.write(data.head())
        
        st.subheader("Data Statistics")
        stats_columns = ['AQI', 'Temperature', 'Humidity', 'Wind_Speed', 'Pressure', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        st.write(data[stats_columns].describe())
        
        st.subheader("AQI Distribution")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['AQI'], kde=True, ax=ax)
            ax.set_title(f'Distribution of AQI Values in {selected_city}')
            ax.set_xlabel('AQI')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Error plotting AQI distribution: {e}")
        
        st.subheader("AQI Time Series")
        try:
            daily_aqi = data.groupby('Date')['AQI'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.plot(daily_aqi['Date'], daily_aqi['AQI'])
            plt.title(f'Daily AQI in {selected_city} Over Time')
            plt.xlabel('Date')
            plt.ylabel('AQI')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Error plotting AQI time series: {e}")
        
        st.subheader("Seasonal AQI Patterns")
        try:
            seasonal_aqi = data.groupby('indian_season')['AQI'].mean().reset_index()
            seasonal_aqi['Season'] = seasonal_aqi['indian_season'].map({
                1: 'Winter', 2: 'Summer', 3: 'Monsoon', 4: 'Post-Monsoon'
            })
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Season', y='AQI', data=seasonal_aqi, ax=ax)
            ax.set_title(f'Average AQI by Indian Season in {selected_city}')
            ax.set_ylabel('Average AQI')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Error plotting seasonal AQI patterns: {e}")
        
        st.subheader("Pollutant Correlations")
        try:
            pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
            corr_matrix = data[pollutants].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix of Pollutants and AQI')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Error plotting pollutant correlations: {e}")
        
        st.subheader("Indian AQI Standards")
        aqi_categories = pd.DataFrame({
            'AQI Range': ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500'],
            'Category': ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'],
            'Health Impact': [
                'Minimal Impact',
                'Minor breathing discomfort to sensitive people',
                'Breathing discomfort to people with lung disease',
                'Breathing discomfort to most people on prolonged exposure',
                'Respiratory illness on prolonged exposure',
                'Serious risk to entire population'
            ]
        })
        st.table(aqi_categories)
    else:
        st.warning(f"No data available for {selected_city}. Please check the dataset.")

# Model Training Page
elif page == "Model Training":
    st.header("Model Training")
    
    if not data.empty:
        X, weather_features, pollutant_features = prepare_features(data, is_scientific_mode)
        y = data['AQI']
        
        if X.shape[1] < 2:
            st.error("Not enough features for training. Ensure dataset has date/time and weather parameters.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.write(f"Training data shape: {X_train.shape}")
            st.write(f"Testing data shape: {X_test.shape}")
            st.write(f"Features used: {', '.join(X.columns.tolist())}")
            
            model_name = st.selectbox("Choose a regression model", ["Random Forest", "Decision Tree", "Linear Regression", "XGBoost"])
            
            if model_name == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
                max_depth = st.slider("Maximum depth", 2, 20, 10, 1)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                normal_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif model_name == "Decision Tree":
                max_depth = st.slider("Maximum depth", 2, 20, 10, 1)
                min_samples_split = st.slider("Minimum samples to split", 2, 10, 2, 1)
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                normal_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            elif model_name == "Linear Regression":
                model = LinearRegression()
                normal_model = LinearRegression()
            elif model_name == "XGBoost":
                n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("Maximum depth", 2, 10, 6, 1)
                model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                normal_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            
            if st.button("Train Model"):
                with st.spinner(f"Training {model_name} in {'Scientific' if is_scientific_mode else 'Normal'} mode..."):
                    if is_scientific_mode:
                        X_normal, _, _ = prepare_features(data, is_scientific_mode=False)
                        X_normal_train, X_normal_test, _, _ = train_test_split(X_normal, y, test_size=0.2, random_state=42)
                        normal_model.fit(X_normal_train, y_train)
                        X_train_weighted = apply_feature_weights(X_train, weather_features, weight=2.0)
                        X_test_weighted = apply_feature_weights(X_test, weather_features, weight=2.0)
                        model.fit(X_train_weighted, y_train)
                        y_pred = model.predict(X_test_weighted)
                        normal_pred = normal_model.predict(X_normal_test)
                        deviation_penalty = 0.1 * (y_pred - normal_pred)**2
                        y_pred = y_pred - deviation_penalty / (1 + np.abs(y_pred))
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        normal_model = model
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{mse:.2f}")
                    col2.metric("RMSE", f"{rmse:.2f}")
                    col3.metric("MAE", f"{mae:.2f}")
                    col4.metric("R²", f"{r2:.2f}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sample_indices = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
                    plt.scatter(y_test.iloc[sample_indices], y_pred[sample_indices], alpha=0.5)
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    plt.xlabel('Actual AQI')
                    plt.ylabel('Predicted AQI')
                    plt.title('Actual vs Predicted AQI')
                    st.pyplot(fig)
                    
                    mode_suffix = "_scientific" if is_scientific_mode else "_normal"
                    model_filename = f"{model_name.lower().replace(' ', '_')}_aqi_model{mode_suffix}.pkl"
                    joblib.dump(model, model_filename)
                    st.success(f"Model saved as {model_filename}")
                    download_model(model, model_name, is_scientific_mode)
                    
                    if model_name in ["Random Forest", "Decision Tree", "XGBoost"]:
                        st.subheader("Feature Importance")
                        importance = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': X_train.columns,
                            'Importance': importance
                        }).sort_values(by='Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                        ax.set_title(f'Feature Importance ({model_name})')
                        st.pyplot(fig)
                        st.table(feature_importance.head(10))
                    
                    if is_scientific_mode:
                        normal_model_filename = f"{model_name.lower().replace(' ', '_')}_aqi_model_normal.pkl"
                        joblib.dump(normal_model, normal_model_filename)
    else:
        st.warning(f"No data available for training in {selected_city}.")

# Make Prediction Page
elif page == "Make Prediction":
    st.header(f"Predict AQI for {selected_city}")
    
    if not data.empty and pm25_model is not None:
        st.subheader("Enter Weather Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            pred_date = st.date_input("Date", datetime.now())
            day_of_week = pred_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            month = pred_date.month
            indian_season_map = {
                12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4
            }
            indian_season = indian_season_map.get(month, 1)
        
        with col2:
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 1.0)
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 2.5, 0.1)
            pressure = st.slider("Pressure (hPa)", 980.0, 1050.0, 1013.0, 0.1)
        
        # Predict PM2.5 for normal mode
        input_data_pm25 = pd.DataFrame({
            'month': [month],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'indian_season': [indian_season],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Wind_Speed': [wind_speed],
            'Pressure': [pressure]
        })
        pm25_pred = predict_pm25(pm25_model, input_data_pm25)
        
        # Pollutant inputs for scientific mode
        pollutant_inputs = {'PM2.5': pm25_pred}
        if is_scientific_mode:
            st.subheader("Enter Pollutant Concentrations")
            col3, col4 = st.columns(2)
            with col3:
                pollutant_inputs['PM2.5'] = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(pm25_pred), 1.0)
                pollutant_inputs['PM10'] = st.slider("PM10 (µg/m³)", 0.0, 600.0, 100.0, 1.0)
                pollutant_inputs['NO2'] = st.slider("NO2 (µg/m³)", 0.0, 200.0, 40.0, 1.0)
            with col4:
                pollutant_inputs['SO2'] = st.slider("SO2 (µg/m³)", 0.0, 100.0, 20.0, 1.0)
                pollutant_inputs['CO'] = st.slider("CO (mg/m³)", 0.0, 10.0, 1.0, 0.1)
                pollutant_inputs['O3'] = st.slider("O3 (µg/m³)", 0.0, 200.0, 30.0, 1.0)
        
        model_name = st.selectbox("Select Trained Model", ["Random Forest", "Decision Tree", "Linear Regression", "XGBoost"])
        mode_suffix = "_scientific" if is_scientific_mode else "_normal"
        model_filename = f"{model_name.lower().replace(' ', '_')}_aqi_model{mode_suffix}.pkl"
        normal_model_filename = f"{model_name.lower().replace(' ', '_')}_aqi_model_normal.pkl"
        
        if not os.path.exists(model_filename):
            st.error(f"No trained {model_name} model found for {'Scientific' if is_scientific_mode else 'Normal'} mode. Please train the model in the 'Model Training' section.")
        elif is_scientific_mode and not os.path.exists(normal_model_filename):
            st.error(f"No trained normal mode {model_name} model found. Please train the model in Scientific mode to generate both models.")
        else:
            model = joblib.load(model_filename)
            normal_model = joblib.load(normal_model_filename) if is_scientific_mode else model
            
            input_data = pd.DataFrame({
                'month': [month],
                'day_of_week': [day_of_week],
                'is_weekend': [is_weekend],
                'indian_season': [indian_season],
                'Temperature': [temperature],
                'Humidity': [humidity],
                'Wind_Speed': [wind_speed],
                'Pressure': [pressure],
                **{k: [v] for k, v in pollutant_inputs.items()}
            })
            
            X_input, weather_features, _ = prepare_features(input_data, is_scientific_mode)
            X_train, _, _ = prepare_features(data, is_scientific_mode)
            input_features = pd.DataFrame(columns=X_train.columns)
            for col in input_features.columns:
                input_features[col] = input_data[col].values if col in input_data.columns else 0
            input_features = input_features[X_train.columns]
            
            if st.button("Predict AQI"):
                try:
                    if is_scientific_mode:
                        input_features_weighted = apply_feature_weights(input_features, weather_features, weight=2.0)
                        prediction = model.predict(input_features_weighted)[0]
                        X_normal, _, _ = prepare_features(input_data, is_scientific_mode=False)
                        normal_features = pd.DataFrame(columns=X_normal.columns)
                        for col in normal_features.columns:
                            normal_features[col] = input_data[col].values if col in input_data.columns else 0
                        normal_pred = normal_model.predict(normal_features)[0]
                        deviation_penalty = 0.1 * (prediction - normal_pred)**2
                        prediction = prediction - deviation_penalty / (1 + abs(prediction))
                    else:
                        prediction = model.predict(input_features)[0]
                        normal_pred = prediction
                    
                    category = "Good" if prediction <= 50 else "Satisfactory" if prediction <= 100 else "Moderate" if prediction <= 200 else "Poor" if prediction <= 300 else "Very Poor" if prediction <= 400 else "Severe"
                    
                    st.subheader("Prediction Results")
                    st.write(f"**Predicted AQI**: {prediction:.2f}")
                    st.write(f"**AQI Category**: {category}")
                    st.write(f"**Predicted PM2.5**: {pm25_pred:.2f} µg/m³")
                    st.write(f"**Health Implications**: {get_health_implications_india(category)}")
                    st.write(f"**Cautionary Statement**: {get_cautionary_statement_india(category)}")
                    if is_scientific_mode:
                        st.write(f"**Normal Mode AQI (Reference)**: {normal_pred:.2f}")
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ranges = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
                    colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8B008B', '#540B0E']
                    for i, (low, high) in enumerate(ranges):
                        ax.fill_between([low, high], [0, 0], [100, 100], color=colors[i], alpha=0.3)
                    ax.axvline(x=prediction, color='blue', linestyle='--', label='Predicted AQI')
                    if is_scientific_mode:
                        ax.axvline(x=normal_pred, color='green', linestyle=':', label='Normal Mode AQI')
                    ax.set_xlim(0, 500)
                    ax.set_ylim(0, 100)
                    ax.set_xlabel('AQI')
                    ax.set_title('AQI Prediction')
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    else:
        st.warning(f"No data or PM2.5 model available for prediction in {selected_city}.")

# AQI Forecast Page
elif page == "AQI Forecast":
    st.header(f"7-Day AQI Forecast for {selected_city}")
    
    if not data.empty and pm25_model is not None:
        model_name = st.selectbox("Select Trained Model", ["Random Forest", "Decision Tree", "Linear Regression", "XGBoost"])
        mode_suffix = "_scientific" if is_scientific_mode else "_normal"
        model_filename = f"{model_name.lower().replace(' ', '_')}_aqi_model{mode_suffix}.pkl"
        normal_model_filename = f"{model_name.lower().replace(' ', '_')}_aqi_model_normal.pkl"
        
        if not os.path.exists(model_filename):
            st.error(f"No trained {model_name} model found for {'Scientific' if is_scientific_mode else 'Normal'} mode. Please train the model in the 'Model Training' section.")
        elif is_scientific_mode and not os.path.exists(normal_model_filename):
            st.error(f"No trained normal mode {model_name} model found. Please train the model in Scientific mode to generate both models.")
        else:
            model = joblib.load(model_filename)
            normal_model = joblib.load(normal_model_filename) if is_scientific_mode else model
            
            forecast_days = st.slider("Number of forecast days", 1, 14, 7)
            forecast_df = generate_forecast(model, normal_model, pm25_model, days=forecast_days, city=selected_city, is_scientific_mode=is_scientific_mode)
            
            st.subheader("Forecast Results")
            st.dataframe(forecast_df)
            
            st.subheader("AQI Forecast Trend")
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(x='Date', y='Predicted AQI', data=forecast_df, marker='o', ax=ax)
                ax.set_title(f'AQI Forecast for {selected_city} ({'Scientific' if is_scientific_mode else 'Normal'} Mode)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Predicted AQI')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Error plotting forecast trend: {e}")
            
            st.subheader("Export Forecast")
            export_to_csv(forecast_df)
            
            st.subheader("Detailed Forecast Information")
            for _, row in forecast_df.iterrows():
                with st.expander(f"{row['Date']} ({row['Day']})"):
                    st.write(f"**Predicted AQI**: {row['Predicted AQI']:.2f}")
                    st.write(f"**Category**: {row['Category']}")
                    st.write(f"**Temperature**: {row['Temperature']}°C")
                    st.write(f"**Humidity**: {row['Humidity']}%")
                    st.write(f"**Wind Speed**: {row['Wind Speed']} m/s")
                    st.write(f"**PM2.5**: {row['PM2.5']} µg/m³")
                    if is_scientific_mode:
                        st.write(f"**PM10**: {row['PM10']} µg/m³")
                        st.write(f"**NO2**: {row['NO2']} µg/m³")
                        st.write(f"**SO2**: {row['SO2']} µg/m³")
                        st.write(f"**CO**: {row['CO']} mg/m³")
                        st.write(f"**O3**: {row['O3']} µg/m³")
                    st.write(f"**Health Implications**: {get_health_implications_india(row['Category'])}")
                    st.write(f"**Cautionary Statement**: {get_cautionary_statement_india(row['Category'])}")
    else:
        st.warning(f"No data or PM2.5 model available for forecasting in {selected_city}.")

# Footer
st.markdown("""
---
Developed with ❤️ using Streamlit | Data Source: [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
""")