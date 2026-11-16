"""LLM pre-entrenado (Hugging Face, por pipeline de transformers) que convierte la predicción
de la red neuronal en un texto con recomendaciones para manejar el pedido."""
import os

import pandas as pd

from features import construir_features
from rl_feedback import ACCIONES

MODELO_LLM = os.environ.get("MODELO_LLM", "Qwen/Qwen2.5-3B-Instruct")

TRADUCCION = {
    "Sunny": "soleado", "Cloudy": "nublado", "Fog": "niebla", "Sandstorms": "tormenta de arena",
    "Stormy": "tormenta", "Windy": "ventoso", "Low": "bajo", "Medium": "medio", "High": "alto",
    "Jam": "atasco", "motorcycle": "moto", "scooter": "scooter", "van": "furgoneta",
    "Metropolitian": "metropolitana", "Urban": "urbana", "Semi-Urban": "semi-urbana", "Other": "otra",
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles", "Thursday": "jueves",
    "Friday": "viernes", "Saturday": "sábado", "Sunday": "domingo",
}

ROL_SISTEMA = (
    "Eres un experto en logística de última milla de una empresa de comercio electrónico en India. "
    "Un modelo de redes neuronales ya predijo cómo llegará un pedido y tú debes escribir recomendaciones "
    "prácticas para manejarlo. Reglas: responde SIEMPRE en español; NO contradigas la predicción; "
    "empieza mencionando el TIPO DE ENTREGA predicho (Rápida, Intermedia o Lenta) y los minutos estimados; "
    "el producto (Clothing, Electronics, Grocery...) es solo lo que se envía, NUNCA es el tipo de entrega; "
    "es una entrega a domicilio: el cliente no recoge el pedido; no inventes datos que no estén "
    "en el pedido; da acciones concretas para la tienda, el repartidor y servicio al cliente; "
    "si la entrega es Rápida no hables de demoras ni ofrezcas descuentos o compensaciones; "
    "no escribas firmas ni plantillas con corchetes; máximo 180 palabras."
)


def cargar_llm(modelo=MODELO_LLM):
    import torch
    from transformers import pipeline
    gpu = torch.cuda.is_available()
    return pipeline("text-generation", model=modelo,
                    torch_dtype=torch.float16 if gpu else torch.float32,
                    device_map="auto" if gpu else None)


def estado_de(resultado):
    """Texto de la red -> estado del agente RL (Rápida / Intermedia / Lenta)."""
    r = str(resultado)
    for estado in ["Rápida", "Intermedia", "Lenta"]:
        if r.startswith(estado):
            return estado
    return None


def _t(v):
    v = str(v).strip()
    return TRADUCCION.get(v, v)


def describir_prediccion(pred):
    estado = estado_de(pred["resultado"])
    lineas = [f"- Tipo de entrega predicho: {pred['resultado']}",
              f"- Tiempo estimado: unos {pred['minutos_estimados']:.0f} minutos (aproximado, ± 18 min)"]
    if estado == "Intermedia":
        lado = "lenta" if pred["prob_lenta"] >= 0.5 else "rápida"
        lineas.append(f"- Ni claramente rápida ni claramente lenta; se inclina hacia {lado} "
                      f"(indicador {pred['prob_lenta']:.0%} hacia lenta)")
    else:
        lineas.append(f"- Probabilidad de entrega lenta (180 min o más): {pred['prob_lenta']:.0%}")
    if estado == "Rápida":
        lineas.append("- Usa el tipo de entrega Rápida como dato principal; los minutos son solo una referencia")
    return "\n".join(lineas)


def describir_pedido(fila_raw, pred):
    f = construir_features(pd.DataFrame([fila_raw])).iloc[0]
    return (
        "DATOS DEL PEDIDO:\n"
        f"- Producto enviado (no es el tipo de entrega): {fila_raw['Category']}\n"
        f"- Distancia tienda-cliente: {f['distance_km']:.1f} km, zona {_t(fila_raw['Area'])}\n"
        f"- Pedido a las {str(fila_raw['Order_Time'])[:5]} del {_t(f['Dia_semana'])}, recogida a las "
        f"{str(fila_raw['Pickup_Time'])[:5]} ({f['espera_pickup_min']:.0f} min de espera)\n"
        f"- Clima: {_t(fila_raw['Weather'])} | Tráfico: {_t(fila_raw['Traffic'])} | "
        f"Vehículo: {_t(fila_raw['Vehicle'])}\n"
        f"- Repartidor: {float(fila_raw['Agent_Age']):.0f} años, calificación {float(fila_raw['Agent_Rating']):.1f}/5\n\n"
        "PREDICCIÓN DEL MODELO NEURONAL:\n" + describir_prediccion(pred)
    )


def construir_prompt(fila_raw, pred, accion):
    usuario = (
        describir_pedido(fila_raw, pred) +
        f"\n\nESTRATEGIA PRINCIPAL (escogida por el agente de aprendizaje por refuerzo): {ACCIONES[accion]}.\n"
        "Escribe la recomendación así:\n"
        "1. Una frase con el diagnóstico (tipo de entrega, minutos y la causa más probable según los datos).\n"
        "2. Tres o cuatro acciones concretas en viñetas, empezando por la estrategia principal.\n"
        "3. Un mensaje corto y amable para el cliente.\n"
        "No agregues más puntos después del 3."
    )
    return [{"role": "system", "content": ROL_SISTEMA}, {"role": "user", "content": usuario}]


def es_coherente(texto, pred, producto=""):
    """La respuesta debe nombrar el tipo de entrega predicho y los minutos estimados, y no confundir
    el producto con el tipo de entrega (p. ej. "La categoría de entrega es Clothing")."""
    raiz = {"Rápida": "rápid", "Intermedia": "intermedi", "Lenta": "lent"}.get(estado_de(pred["resultado"]), "")
    t = texto.lower().replace("rapid", "rápid")
    primera_frase = t.split("\n")[0].split(".")[0]
    producto = str(producto).strip().lower()
    confunde = bool(producto) and producto in primera_frase and (
        "categoría" in primera_frase or "tipo de entrega" in primera_frase)
    return raiz in t and f"{pred['minutos_estimados']:.0f}" in texto and not confunde


def generar_descripcion_llm(llm, fila_raw, pred, accion, max_tokens=350, temperatura=0.4, intentos=3):
    mensajes = construir_prompt(fila_raw, pred, accion)
    texto = ""
    for _ in range(intentos):
        salida = llm(mensajes, max_new_tokens=max_tokens, do_sample=True,
                     temperature=temperatura, top_p=0.9)
        texto = salida[0]["generated_text"][-1]["content"].strip()
        if es_coherente(texto, pred, fila_raw["Category"]):
            break
    return texto
