# %%
import gradio as gr
import pandas as pd
import joblib
import numpy as np
import geopy.distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Laden und Vorbereiten der Daten

# %%
# Laden der Apartment-Daten
df = pd.read_csv("apartments_data_enriched_lat_lon_combined.csv")

# Bestimmen der korrekten Spaltennamen für Koordinaten
lat_col = "latitude" if "latitude" in df.columns else "lat"
lon_col = "longitude" if "longitude" in df.columns else "lon"

# Sicherstellen, dass die Spalten existieren
if lat_col not in df.columns or lon_col not in df.columns:
    raise KeyError("Fehlende Spalten für Koordinaten: Erwartet 'latitude' oder 'lat' und 'longitude' oder 'lon'")

# Beispielhafte öffentliche Verkehrsmittel-Stationen
public_transit_stations = [
    {"name": "Hauptbahnhof", "lat": 47.378177, "lon": 8.540192},
    {"name": "Bahnhof Stadelhofen", "lat": 47.366321, "lon": 8.548008},
    {"name": "Hardbrücke", "lat": 47.385118, "lon": 8.517220},
    {"name": "Enge", "lat": 47.364751, "lon": 8.531601}
]

# %% [markdown]
# ## Feature Engineering: Entfernung zu ÖV-Stationen

# %%
# Funktion zur Berechnung der Distanz zur nächsten Haltestelle
def distance_to_nearest_station(lat, lon):
    return min(
        geopy.distance.geodesic((lat, lon), (station["lat"], station["lon"])).km
        for station in public_transit_stations
    )

# Berechnung der Entfernung zur nächsten ÖV-Station
df["distance_to_transit"] = df.apply(lambda row: distance_to_nearest_station(row[lat_col], row[lon_col]), axis=1)

# %% [markdown]
# ## Modelltraining

# %%
# Definierte Features und Zielvariable
features = ["rooms", "area", "pop_dens", "tax_income", "distance_to_transit"]
X = df[features]
y = df["price"]

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trainieren des RandomForest-Modells
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Speichern des Modells
joblib.dump(model, "apartment_price_model.pkl")

# %% [markdown]
# ## Gradio Interface für Preisvorhersage

# %%
# Vorhersagefunktion für Gradio
def predict_price(rooms, area, pop_dens, tax_income, distance_to_transit):
    input_data = pd.DataFrame([[rooms, area, pop_dens, tax_income, distance_to_transit]], columns=features)
    prediction = model.predict(input_data)[0]
    return f"Geschätzter Preis: {prediction:.2f} CHF"

# Gradio Interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Anzahl Zimmer"),
        gr.Number(label="Fläche (m²)"),
        gr.Number(label="Bevölkerungsdichte"),
        gr.Number(label="Steuerbares Einkommen"),
        gr.Number(label="Distanz zur nächsten ÖV-Station (km)")
    ],
    outputs="text",
    title="Wohnungspreis-Vorhersage",
    description="Geben Sie die Merkmale einer Wohnung ein, um den geschätzten Preis zu berechnen."
)

demo.launch()