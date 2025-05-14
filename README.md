# 🌬️ AQI Predictor AI/ML

> Empowering Communities with Accurate Air Quality Predictions

## 📌 Project Overview

The AQI Predictor AI/ML is a sophisticated machine learning web application designed to forecast the Air Quality Index (AQI) for Indian cities. Built using Streamlit and powered by advanced ML models, it provides real-time AQI predictions based on weather and pollutant data. The app supports two modes: Normal (Weather + PM2.5) and Scientific (Weather + Pollutants), catering to both general users and environmental researchers. With an interactive dashboard, users can explore historical data, train models, make predictions, and generate multi-day AQI forecasts.

## 🎯 Objectives

- **Accurate AQI Forecasting**: Predict AQI using models like Random Forest, XGBoost, Decision Tree, and Linear Regression.
- **Dual-Mode Predictions**: Offer simplified (Normal) and detailed (Scientific) prediction modes.
- **Data-Driven Insights**: Visualize trends, correlations, and seasonal patterns in air quality data.
- **User Accessibility**: Provide an intuitive interface for all users, from individuals to policymakers.

## 🌟 Features

### Dual Prediction Modes:
- **Normal Mode**: Uses weather data and predicted PM2.5 for simplified AQI forecasting.
- **Scientific Mode**: Incorporates multiple pollutants (PM2.5, PM10, NO2, SO2, CO, O3) for detailed predictions.

### Multiple ML Models: 
Supports Random Forest, Decision Tree, Linear Regression, and XGBoost with customizable hyperparameters.

### Interactive Data Exploration:
- Visualize AQI distributions, time-series trends, seasonal patterns, and pollutant correlations.
- Display Indian AQI standards with health implications.

### Additional Features:
- **Real-Time Predictions**: Input weather and pollutant parameters to predict AQI instantly.
- **7-Day Forecasts**: Generate multi-day AQI forecasts with weather and pollutant estimates.
- **Model Export**: Download trained models and forecast data as CSV files.
- **City-Specific Analysis**: Supports multiple Indian cities from the dataset.

## 🚀 Demo

Try the AQI Predictor live!  
👉 **Deployed App**: [howzmyair.streamlit.app](https://howzmyair.streamlit.app)

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning Libraries**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Dataset**: Air Quality Data in India (Kaggle)
- **Deployment**: Streamlit Cloud/Heroku
- **Version Control**: Git, GitHub

## 📊 How It Works

1. **Data Loading**: Loads city-specific air quality data from city_day.csv (Kaggle dataset).
2. **Preprocessing**: 
   - Generates synthetic weather data (Temperature, Humidity, Wind Speed, Pressure) based on monthly averages.
   - Creates time-based features (month, day of week, Indian seasons).
   - Fills missing pollutant and AQI values using median or PM2.5-based estimates.

3. **Model Training**:
   - Trains a PM2.5 prediction model (Random Forest) for Normal mode.
   - Trains user-selected models (Random Forest, Decision Tree, Linear Regression, XGBoost) for AQI prediction.
   - Applies feature weighting in Scientific mode to emphasize weather factors.

4. **Prediction**:
   - **Normal Mode**: Predicts PM2.5 and uses it with weather data for AQI prediction.
   - **Scientific Mode**: Uses user-input pollutants and weather data, with a penalty for deviation from Normal mode predictions.

5. **Forecasting**: Generates 7-day AQI forecasts using estimated weather and pollutant trends.
6. **Visualization**: Displays AQI trends, feature importance, and correlations via interactive plots.

## 🖥️ Installation and Setup

Follow these steps to run the project locally:

### Prerequisites

- Python 3.8+
- Git
- Virtualenv (recommended)
- Dataset: Download city_day.csv from Kaggle and place it in the project directory.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VishnuKC26/aqi-predictor-ai-ml.git
   cd aqi-predictor-ai-ml
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure requirements.txt includes: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib.

4. **Add Dataset**: Place city_day.csv in the project root directory.

5. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

6. **Access the App**: Open your browser and navigate to http://localhost:8501.

## 📈 Model Performance

The app evaluates multiple models for AQI prediction. Example performance:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Random Forest | 0.92 | 15.2 | 20.1 |
| XGBoost | 0.94 | 13.8 | 18.5 |
| Decision Tree | 0.87 | 18.5 | 24.3 |
| Linear Regression | 0.78 | 22.1 | 28.7 |

*Note: Metrics depend on the dataset and hyperparameters. Train models in the app to view actual results.*

## 🌍 Use Cases

- **Public Health**: Helps residents plan outdoor activities based on AQI forecasts.
- **Environmental Research**: Enables analysis of pollutant trends and correlations.
- **Urban Planning**: Supports cities in monitoring and improving air quality.
- **Education**: Demonstrates ML applications in environmental science.

## 🤝 Contributing

Contributions are welcome to enhance the AQI Predictor! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

Please read our Contributing Guidelines for details.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgments

- **Dataset**: Air Quality Data in India by Rohan Rao.
- **Inspiration**: Open-source ML projects and environmental research initiatives.
- **Built by**: VishnuKC26.

## 📬 Contact

Have questions or suggestions? Reach out!  

- **GitHub**: [VishnuKC26](https://github.com/VishnuKC26)  
- **Email**: vishnukc26@example.com

---

⭐ Star this repository if you find it useful! Let's work together for cleaner air. 🌱
