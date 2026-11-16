import numpy as np
import pandas as pd

CAT_COLS = ["Weather", "Traffic", "Vehicle", "Area", "Category", "Dia_semana"]
NUM_COLS = ["distance_km", "Agent_Age", "Agent_Rating",
            "hora_decimal", "hora_sin", "hora_cos",
            "espera_pickup_min", "dia_mes",
            "Store_Latitude", "Store_Longitude"]
FEATURES = NUM_COLS + CAT_COLS

# Columnas "crudas" que necesita el formulario / el CSV para poder construir las variables
RAW_COLS = ["Agent_Age", "Agent_Rating",
            "Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude",
            "Order_Date", "Order_Time", "Pickup_Time",
            "Weather", "Traffic", "Vehicle", "Area", "Category"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def limpiar_texto(serie):
    s = serie.astype(str).str.strip()
    return s.replace({"NaN": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})


def parse_hora(serie):
    # acepta "11:30:00" y también "11:30" (como lo escribe una persona en un formulario)
    s = serie.astype(str).str.strip()
    s = s.where(s.str.count(":") != 1, s + ":00")
    return pd.to_datetime(s, format="%H:%M:%S", errors="coerce")


def construir_features(df_raw):
    d = df_raw.copy()

    for c in ["Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude",
              "Agent_Age", "Agent_Rating"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"]:
        d[c] = d[c].abs()

    d["distance_km"] = haversine_km(d["Store_Latitude"], d["Store_Longitude"],
                                    d["Drop_Latitude"], d["Drop_Longitude"])

    t_pedido = parse_hora(d["Order_Time"])
    t_pickup = parse_hora(d["Pickup_Time"])
    min_pedido = t_pedido.dt.hour * 60 + t_pedido.dt.minute
    min_pickup = t_pickup.dt.hour * 60 + t_pickup.dt.minute

    d["hora_decimal"] = min_pedido / 60.0
    d["hora_sin"] = np.sin(2 * np.pi * d["hora_decimal"] / 24.0)
    d["hora_cos"] = np.cos(2 * np.pi * d["hora_decimal"] / 24.0)
    d["espera_pickup_min"] = (min_pickup - min_pedido) % 1440

    fecha = pd.to_datetime(d["Order_Date"], errors="coerce")
    d["Dia_semana"] = fecha.dt.day_name()
    d["dia_mes"] = fecha.dt.day

    for c in ["Weather", "Traffic", "Vehicle", "Area", "Category"]:
        d[c] = limpiar_texto(d[c])

    return d[FEATURES]
