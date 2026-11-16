"""API web: formulario -> red neuronal -> agente RL -> LLM -> recomendación -> feedback -> tabla Q.

Correr local:   uvicorn api:app --host 0.0.0.0 --port 8000     ->  http://127.0.0.1:8000
"""
import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from features import RAW_COLS
from recomendador import MODELO_LLM, cargar_llm, estado_de, generar_descripcion_llm
from rl_feedback import ACCIONES, AgenteQ

CARPETA = os.path.dirname(os.path.abspath(__file__))


class Pedido(BaseModel):
    Agent_Age: float
    Agent_Rating: float
    Store_Latitude: float
    Store_Longitude: float
    Drop_Latitude: float
    Drop_Longitude: float
    Order_Date: str
    Order_Time: str
    Pickup_Time: str
    Weather: str
    Traffic: str
    Vehicle: str
    Area: str
    Category: str


class Feedback(BaseModel):
    estado: str
    accion: str
    estrellas: int | None = None      # calificación de la recomendación del LLM (1 a 5)
    me_gusta: bool | None = None      # me gusta / no me gusta de una acción del ranking
    comentario: str = ""


def crear_app(predictor=None, llm=None, agente=None):
    if predictor is None:
        from inferencia import Predictor
        predictor = Predictor(CARPETA)
    if agente is None:
        agente = AgenteQ(os.path.join(CARPETA, "q_tabla.json"))
    if llm is None:
        llm = cargar_llm()

    app = FastAPI(title="Proyecto 8: red neuronal + LLM + aprendizaje por refuerzo")
    # Permite que la página publicada en Vercel (otro dominio) llame a esta API en Colab.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    def inicio():
        return FileResponse(os.path.join(CARPETA, "static", "index.html"))

    @app.get("/salud")
    def salud():
        return {"ok": True, "modelo_neuronal": predictor.cfg["modelo"], "llm": MODELO_LLM,
                "epsilon": round(agente.epsilon, 3), "feedbacks": len(agente.historial)}

    @app.get("/opciones")
    def opciones():
        return {"opciones": predictor.cfg["opciones"], "acciones": ACCIONES}

    @app.post("/predict")
    def predict(pedido: Pedido):
        fila = pd.Series(pedido.model_dump())[RAW_COLS]
        pred = predictor.predecir(pd.DataFrame([fila])).iloc[0]
        estado = estado_de(pred["resultado"])
        if estado is None or pd.isna(pred["minutos_estimados"]):
            raise HTTPException(422, f"{pred['resultado']}. {pred.get('motivo', '')}".strip())

        accion, modo = agente.elegir_accion(estado)
        t0 = time.time()
        texto = generar_descripcion_llm(llm, fila, pred, accion)
        ranking = [{"accion": a, "texto": ACCIONES[a], "Q": round(agente.Q[estado][a], 3)}
                   for a in agente.ranking(estado) if a != accion][:3]   # sin repetir la ya escogida
        return {
            "prediccion": {"resultado": pred["resultado"], "minutos_estimados": round(float(pred["minutos_estimados"]), 1),
                           "prob_lenta": round(float(pred["prob_lenta"]), 4)},
            "estado": estado, "accion": accion, "accion_texto": ACCIONES[accion], "modo": modo,
            "epsilon": round(agente.epsilon, 3),
            "recomendacion_llm": texto, "segundos_llm": round(time.time() - t0, 1),
            "ranking_rl": ranking,
        }

    @app.post("/feedback")
    def feedback(fb: Feedback):
        if fb.estado not in agente.Q or fb.accion not in ACCIONES:
            raise HTTPException(400, "estado o acción desconocidos")
        if fb.estrellas is not None:
            r, q = agente.feedback_estrellas(fb.estado, fb.accion, fb.estrellas, fb.comentario)
        elif fb.me_gusta is not None:
            r, q = agente.feedback_me_gusta(fb.estado, fb.accion, fb.me_gusta)
        else:
            raise HTTPException(400, "envía 'estrellas' o 'me_gusta'")
        return {"recompensa": r, "q_nuevo": round(q, 4), "epsilon": round(agente.epsilon, 3),
                "feedbacks": len(agente.historial), "tabla_q": agente.tabla()}

    @app.get("/q_tabla")
    def q_tabla():
        return {"epsilon": round(agente.epsilon, 3), "feedbacks": len(agente.historial),
                "tabla_q": agente.tabla()}

    app.state.agente, app.state.predictor, app.state.llm = agente, predictor, llm
    return app


# En el notebook se pone API_AUTOLOAD=0 para crear la app con los objetos ya cargados.
if os.environ.get("API_AUTOLOAD", "1") == "1":
    app = crear_app()
