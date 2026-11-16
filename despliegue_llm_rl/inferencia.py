import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from features import construir_features


def calibrar(p, T):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-z / T))


class Predictor:
    def __init__(self, carpeta="."):
        self.clf = tf.keras.models.load_model(f"{carpeta}/clasificador.keras")
        self.reg = tf.keras.models.load_model(f"{carpeta}/regresor_minutos.keras")
        self.prep = joblib.load(f"{carpeta}/preprocesador.joblib")
        with open(f"{carpeta}/config.json", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def predecir(self, df_raw):
        c = self.cfg
        feats = construir_features(df_raw.reset_index(drop=True))
        validas = ~feats.isna().any(axis=1).to_numpy()
        res = pd.DataFrame({"prob_lenta": np.nan, "minutos_estimados": np.nan,
                            "resultado": "Datos inválidos"}, index=feats.index)
        if validas.any():
            X = self.prep.transform(feats[validas]).astype("float32")
            prob = calibrar(self.clf.predict(X, verbose=0).ravel(), c["temperatura"])
            minutos = self.reg.predict(X, verbose=0).ravel()
            etiqueta = np.where(prob >= c["umbral_decision"], c["nombres"]["lenta"], c["nombres"]["rapida"])
            intermedia = (minutos > c["corte_intermedia_A"]) & (minutos < c["corte_intermedia_B"])
            etiqueta = np.where(intermedia, c["nombres"]["intermedia"], etiqueta)
            res.loc[validas, "prob_lenta"] = prob
            res.loc[validas, "minutos_estimados"] = minutos
            res.loc[validas, "resultado"] = etiqueta
        return res
