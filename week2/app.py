import joblib
import pandas as pd
import numpy as np
import gradio as gr
import geopy.distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ------------------------------
# Load Data
# ------------------------------
DATA_PATH = "apartments_data_enriched_lat_lon_combined.csv"
df = pd.read_csv(DATA_PATH)

# Handle missing coordinate columns
lat_col = "latitude" if "latitude" in df.columns else "lat"
lon_col = "longitude" if "longitude" in df.columns else "lon"
if lat_col not in df.columns or lon_col not in df.columns:
    raise KeyError("Latitude and Longitude columns are missing!")

# Define public transport stations
public_transit_stations = [
    {"name": "Hauptbahnhof", "lat": 47.378177, "lon": 8.540192},
    {"name": "Bahnhof Stadelhofen", "lat": 47.366321, "lon": 8.548008},
    {"name": "Hardbrücke", "lat": 47.385118, "lon": 8.517220},
    {"name": "Enge", "lat": 47.364751, "lon": 8.531601}
]

# Function to compute distance to nearest station
def distance_to_nearest_station(lat, lon):
    min_distance = np.inf
    for station in public_transit_stations:
        dist = geopy.distance.geodesic((lat, lon), (station["lat"], station["lon"]))
        min_distance = min(min_distance, dist.km)
    return min_distance

# Compute new feature
df["distance_to_transit"] = df.apply(lambda row: distance_to_nearest_station(row[lat_col], row[lon_col]), axis=1)

# ------------------------------
# Train Model
# ------------------------------
features = ["rooms", "area", "pop_dens", "tax_income", "distance_to_transit"]
X = df[features]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "apartment_price_model.pkl")

# ------------------------------
# Gradio Web Interface
# ------------------------------
def predict_price(rooms, area, pop_dens, tax_income, distance_to_transit):
    model = joblib.load("apartment_price_model.pkl")
    input_data = pd.DataFrame([[rooms, area, pop_dens, tax_income, distance_to_transit]], columns=features)
    prediction = model.predict(input_data)[0]
    return f"Geschätzter Preis: {prediction:.2f} CHF"

app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Anzahl Zimmer"),
        gr.Number(label="Fläche (m²)"),
        gr.Number(label="Bevölkerungsdichte"),
        gr.Number(label="Steuerbares Einkommen"),
        gr.Number(label="Distanz zur nächsten ÖV-Station (km)")
    ],
    outputs="text"
)

if __name__ == "__main__":
    app.launch()
