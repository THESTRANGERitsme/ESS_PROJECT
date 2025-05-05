import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------ Data Loading ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Forest_Area.csv")
    # Clean the "Country and Area" column
    df["Country and Area"] = df["Country and Area"].astype(str).str.strip()
    
    # Convert historical forest area columns to numeric
    historical_cols = ["Forest Area, 1990", "Forest Area, 2000", "Forest Area, 2010",
                       "Forest Area, 2015", "Forest Area, 2020"]
    for col in historical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# ------------------ Main App ------------------
st.title("Future Forest Deforestation Predictor")
st.write("Enter a country name and a future year to predict the forest area based on all past data and determine if deforestation (decline in forest area) is expected.")

# Inputs: Country name and future year
country_input = st.text_input("Country Name:")
future_year = st.number_input("Future Year (must be > 2020):", min_value=2021, value=2050, step=1)

if st.button("Predict"):
    if not country_input:
        st.error("Please enter a country name.")
    else:
        # Filter by country (case-insensitive)
        country_filter = df["Country and Area"].str.lower().str.contains(country_input.lower(), na=False)
        filtered = df[country_filter]
        if filtered.empty:
            st.error("No matching country found.")
        else:
            # Use the first matching record.
            record = filtered.iloc[0]
            st.write("Historical Data for", record["Country and Area"])
            
            # Historical years and forest area values
            years = np.array([1990, 2000, 2010, 2015, 2020]).reshape(-1, 1)
            forest_areas = np.array([
                record["Forest Area, 1990"],
                record["Forest Area, 2000"],
                record["Forest Area, 2010"],
                record["Forest Area, 2015"],
                record["Forest Area, 2020"]
            ])
            
            # Use only valid (non-NaN) data points.
            valid_mask = ~np.isnan(forest_areas)
            years_valid = years[valid_mask]
            forest_areas_valid = forest_areas[valid_mask]
            
            if len(forest_areas_valid) < 2:
                st.error("Not enough data points to make a prediction.")
            else:
                # Build linear regression using all valid historical data.
                model = LinearRegression()
                model.fit(years_valid, forest_areas_valid)
                
                # Predict forest area for the future year.
                predicted_area = model.predict(np.array([[future_year]]))[0]
                
                # Determine trend based on regression slope:
                # If the slope is negative, the forest area is decreasing (deforestation).
                trend = "Yes" if model.coef_[0] < 0 else "No"
                
                st.subheader(f"Prediction for {record['Country and Area']}:")
                st.write(f"Predicted Forest Area in {future_year}: {predicted_area:.2f}")
                st.write("Will deforestation (decline in forest area) occur based on past trends?", trend)
                
                # ------------------ Plotting ------------------
                fig, ax = plt.subplots()
                # Plot historical data points.
                ax.scatter(years_valid, forest_areas_valid, color='blue', label="Historical Data")
                # Create a regression line from the earliest historical year to the future year.
                x_range = np.linspace(years_valid.min(), future_year, 100).reshape(-1, 1)
                y_range = model.predict(x_range)
                ax.plot(x_range, y_range, color='red', label="Regression Line")
                # Mark the predicted future point.
                ax.scatter(future_year, predicted_area, color='green', s=100, label="Predicted Value")
                ax.set_xlabel("Year")
                ax.set_ylabel("Forest Area")
                ax.set_title(f"Forest Area Trend for {record['Country and Area']}")
                ax.legend()
                st.pyplot(fig)
