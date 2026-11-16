"""Genera Proyecto_8_Corte3_LLM_RL.ipynb (Google Colab, GPU T4) a partir de los archivos de despliegue_llm_rl/."""
import json
from pathlib import Path

AQUI = Path(__file__).parent
DEP = AQUI / "despliegue_llm_rl"
celdas = []


def md(texto):
    celdas.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n")})


def code(texto):
    celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": texto.strip("\n")})


def writefile(ruta_rel):
    contenido = (DEP / ruta_rel).read_text(encoding="utf-8")
    code(f"%%writefile despliegue_llm_rl/{ruta_rel}\n" + contenido)


# ---------------------------------------------------------------------------
md("""
# Proyecto 8 — Corte 3 Final: Modelo neuronal + LLM + Aprendizaje por Refuerzo

**Integrantes:** Kevin Tapias, Sebastian Villera y Anuar Marriaga

```
 Formulario web ─► Red neuronal (Corte 2) ─► predicción ─┐
 (index.html)       Rápida / Intermedia / Lenta,          │
                    minutos y prob. de entrega lenta      ▼
                                       Agente Q-learning escoge la estrategia (acción)
                                                          │
                                                          ▼
                              Prompt (rol + datos + predicción + estrategia) ─► LLM Qwen2.5 ─► Recomendación
                                                                                                   │
                         Tabla Q se actualiza ◄── recompensa / castigo ◄── usuario califica (★ / 👍 👎)
```

1. **Modelo neuronal (Corte 2):** el ensamble de 6 redes + regresor de minutos predice si el pedido llegará **Rápido (≤70 min)**, **Intermedio** o **Lento (≥180 min)**, con los minutos estimados y la probabilidad de entrega lenta.
2. **LLM pre-entrenado (Qwen2.5-3B-Instruct, Hugging Face)** conectado por **pipeline** de `transformers`: la predicción entra como **input** del prompt y el LLM redacta recomendaciones para manejar el pedido.
3. **Aprendizaje por refuerzo (Q-learning):** el usuario **califica** la recomendación; la calificación es la **recompensa o castigo** que actualiza la **tabla Q** (estados × acciones). Con ε-greedy el agente pasa de **exploración** a **explotación**.
4. **Despliegue web:** API en **FastAPI** (`/predict`, `/feedback`, `/q_tabla`) + página `index.html` con formulario y calificación.

> Abrir en **Google Colab** → Entorno de ejecución → Cambiar tipo → **GPU T4**. Correr celda por celda.
""")

md("""
## Paso 1: instalar y cargar librerías

- `transformers` + `accelerate`: descargar y correr el LLM pre-entrenado.
- `fastapi` + `uvicorn`: la API web.
- `scikit-learn==1.6.1`: la misma versión con la que se guardó `preprocesador.joblib` en el Corte 2.
""")
code("""
!pip install -q "scikit-learn==1.6.1" transformers accelerate fastapi uvicorn
""")
code("""
import os, sys, json, time, zipfile, shutil, importlib, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

random.seed(42); np.random.seed(42); tf.keras.utils.set_random_seed(42); torch.manual_seed(42)
print("TensorFlow:", tf.__version__, "| PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO HAY GPU (activar T4)")
""")

md("""
## Paso 2: subir el modelo del Corte 2 y el dataset

Subir **dos archivos**:
- `despliegue.zip` (o `despliegue (1).zip`): `clasificador.keras`, `regresor_minutos.keras`, `preprocesador.joblib`, `features.py`, `inferencia.py` y `config.json` del Corte 2.
- `amazon_delivery.csv`: para tomar pedidos reales y probar el sistema.

Todo queda en la carpeta `despliegue_llm_rl/`, que es la aplicación final.
""")
code("""
from google.colab import files

CARPETA = "despliegue_llm_rl"
os.makedirs(f"{CARPETA}/static", exist_ok=True)

import glob
# Se pueden subir los dos a la vez (Ctrl + clic) o de a uno corriendo esta celda dos veces:
# los archivos se buscan en el disco, no solo en lo último que se subió.
faltan = lambda: not glob.glob("*.csv") or not [z for z in glob.glob("*.zip") if z != "despliegue_llm_rl.zip"]
if faltan():
    files.upload()
if faltan():
    raise SystemExit("Falta subir el .zip del modelo o el .csv: vuelve a correr esta celda y súbelo.")

CSV_PATH = sorted(glob.glob("*.csv"))[0]
for nombre in glob.glob("*.zip"):
    if nombre != "despliegue_llm_rl.zip":
        with zipfile.ZipFile(nombre) as z:
            z.extractall("_modelo_corte2")

for raiz, _, archivos in os.walk("_modelo_corte2"):
    for a in archivos:
        if a.endswith((".keras", ".joblib", ".json", ".py")):
            shutil.copy(os.path.join(raiz, a), CARPETA)
print("Archivos del modelo:", sorted(os.listdir(CARPETA)))
print("CSV:", CSV_PATH)
if CARPETA not in sys.path:
    sys.path.insert(0, CARPETA)
""")

md("""
## Paso 3: modelo neuronal — predicción con `model.predict`

`Predictor.predecir` construye las variables con `features.py` (las mismas del entrenamiento), aplica el preprocesador y llama a `model.predict` del clasificador y del regresor. Probamos tres pedidos escritos a mano.
""")
code("""
import features, inferencia
importlib.reload(features); importlib.reload(inferencia)
from features import RAW_COLS

predictor = inferencia.Predictor(CARPETA)
CFG = predictor.cfg
print(f"Modelo: {CFG['modelo']} | acierto test: {CFG['acierto_test']:.4f} | error test: {CFG['error_test']:.4f}")

pedidos_demo = pd.DataFrame([
    {"Agent_Age": 25, "Agent_Rating": 4.9, "Store_Latitude": 12.914, "Store_Longitude": 77.678,
     "Drop_Latitude": 12.934, "Drop_Longitude": 77.688, "Order_Date": "2022-03-15",
     "Order_Time": "10:00", "Pickup_Time": "10:10", "Weather": "Sunny", "Traffic": "Low",
     "Vehicle": "motorcycle", "Area": "Urban", "Category": "Grocery"},
    {"Agent_Age": 30, "Agent_Rating": 4.6, "Store_Latitude": 12.914, "Store_Longitude": 77.678,
     "Drop_Latitude": 12.954, "Drop_Longitude": 77.718, "Order_Date": "2022-03-19",
     "Order_Time": "18:30", "Pickup_Time": "18:45", "Weather": "Fog", "Traffic": "Jam",
     "Vehicle": "motorcycle", "Area": "Metropolitian", "Category": "Clothing"},
    {"Agent_Age": 38, "Agent_Rating": 4.1, "Store_Latitude": 22.745, "Store_Longitude": 75.892,
     "Drop_Latitude": 22.875, "Drop_Longitude": 76.022, "Order_Date": "2022-03-20",
     "Order_Time": "19:30", "Pickup_Time": "19:45", "Weather": "Stormy", "Traffic": "Jam",
     "Vehicle": "scooter", "Area": "Metropolitian", "Category": "Electronics"},
])
pred_demo = predictor.predecir(pedidos_demo)
display(pd.concat([pedidos_demo[["Weather", "Traffic", "Vehicle", "Area", "Order_Time"]], pred_demo], axis=1).round(3))
""")

md("""
## Paso 4: aprendizaje por refuerzo — módulo Q-learning

Lo escribimos antes que el LLM porque el prompt usa la **acción** que escoge el agente. Se explica en detalle en el Paso 7.
""")
writefile("rl_feedback.py")

md("""
## Paso 5: LLM pre-entrenado — Qwen2.5-3B-Instruct

**Descripción del LLM:** usamos **Qwen2.5-3B-Instruct**, un modelo de lenguaje grande (LLM) de código abierto de Alibaba Cloud, publicado en Hugging Face. Es un Transformer tipo *decoder-only* de unos 3 mil millones de parámetros, pre-entrenado con alrededor de 18 billones de tokens en más de 29 idiomas (incluido el español) y ajustado con instrucciones y RLHF para seguir órdenes como un chat. Acepta mensajes con **rol** (`system`, `user`, `assistant`), lo que nos deja darle el papel de experto en logística. Corre en la GPU T4 gratuita de Colab en float16 (~6 GB). **No lo re-entrenamos**: lo conectamos por **pipeline** de `transformers` (`pipeline("text-generation")`), y la predicción de nuestra red neuronal entra como input del prompt. Lo escogimos por ser gratuito, sin API key, liviano y bueno en español; es el mismo tipo de modelo (Qwen) que se mostró en clase. Si se quiere, `MODELO_LLM` se cambia por `Qwen/Qwen2.5-1.5B-Instruct` (más rápido) o `meta-llama/Llama-3.2-3B-Instruct` (pide token de Hugging Face).

**Parámetros de generación:** `max_new_tokens=350`, `temperature=0.4` (poca creatividad, para que no invente) y `top_p=0.9`.
""")
writefile("recomendador.py")
code("""
import rl_feedback, recomendador
importlib.reload(rl_feedback); importlib.reload(recomendador)
from rl_feedback import AgenteQ, ACCIONES, ESTADOS, Q_INICIAL
from recomendador import (MODELO_LLM, cargar_llm, construir_prompt, generar_descripcion_llm,
                          es_coherente, estado_de)

t0 = time.time()
llm = cargar_llm(MODELO_LLM)
print(f"LLM cargado: {MODELO_LLM} | {llm.model.num_parameters()/1e9:.2f} mil millones de parámetros | {time.time()-t0:.0f} s")
""")

md("""
## Paso 6: diseño del prompt — la predicción entra como INPUT al LLM

El mensaje tiene:
1. **Rol del sistema:** *"Eres un experto en logística de última milla…"* con reglas: español, no contradecir la predicción, empezar con el tipo de entrega y los minutos, no confundir el producto con el tipo de entrega, no ofrecer compensaciones si la entrega es Rápida, no inventar datos.
2. **Datos del pedido** (producto, distancia, hora, clima, tráfico, vehículo, repartidor) **+ predicción de la red** (tipo de entrega, minutos y probabilidad).
3. **Estrategia principal**: la acción que escogió el agente de Q-learning.
4. **Formato:** diagnóstico, 3–4 acciones concretas y un mensaje para el cliente.

Además, `es_coherente` revisa que el texto nombre el tipo de entrega predicho y los minutos, y que no diga que el producto es la "categoría"; si no, se genera otra vez (hasta 3 intentos).

Así se ve el prompt real del pedido lento:
""")
code("""
mensajes = construir_prompt(pedidos_demo.iloc[2], pred_demo.iloc[2], "avisar_cliente")
for m in mensajes:
    print(f"[{m['role'].upper()}]\\n{m['content']}\\n")
""")
code("""
for i in range(len(pedidos_demo)):
    fila, pred = pedidos_demo.iloc[i], pred_demo.iloc[i]
    estado = estado_de(pred["resultado"])
    accion = max(Q_INICIAL[estado], key=Q_INICIAL[estado].get)   # mejor acción según la tabla Q inicial
    t0 = time.time()
    texto = generar_descripcion_llm(llm, fila, pred, accion)
    print("=" * 100)
    print(f"PEDIDO {i+1}: {pred['resultado']} | {pred['minutos_estimados']:.0f} min | P(lenta) {pred['prob_lenta']:.0%} "
          f"| acción: {accion} | coherente: {es_coherente(texto, pred, fila['Category'])} | {time.time()-t0:.1f} s")
    print("-" * 100)
    print(texto)
""")

md("""
## Paso 7: Aprendizaje por Refuerzo (Q-learning) con el feedback del usuario

No es un modelo supervisado: es un **agente** con una **tabla Q** de valores discretos.

| Elemento | En nuestro proyecto |
|---|---|
| **Estados (s)** | Lo que predijo la red: `Rápida`, `Intermedia`, `Lenta` |
| **Acciones (a)** | 6 estrategias predeterminadas: `despacho_estandar`, `priorizar_tienda`, `ruta_alternativa`, `reasignar_repartidor`, `avisar_cliente`, `compensar_reprogramar` |
| **Tabla Q** | 3 × 6 = 18 casillas. Arranca con valores predeterminados (conocimiento previo, p. ej. en `Lenta` vale más avisar al cliente que el despacho estándar) |
| **Recompensa / castigo** | Estrellas: r = (★ − 3) / 2 → 1★ = −1, 3★ = 0, 5★ = +1. Me gusta = +1, No me gusta = −1 |
| **Actualización** | Q(s,a) ← Q(s,a) + α · [ r + γ · maxₐ′ Q(s′,a′) − Q(s,a) ], con α = 0.3, γ = 0.9. Cada pedido es un episodio de un paso (s′ terminal), así que el objetivo es r |
| **Política ε-greedy** | Con probabilidad ε explora (acción al azar); si no, explota (mayor Q). ε = 0.5 · 0.95ⁿ (mínimo 0.05): con cada feedback explora menos |

La tabla se guarda en `q_tabla.json`, así el aprendizaje sigue entre sesiones y lo comparten el notebook y la página web.
""")
code("""
agente = AgenteQ(f"{CARPETA}/q_tabla.json", alfa=0.3, gamma=0.9, epsilon_inicial=0.5,
                 epsilon_min=0.05, decaimiento=0.95, semilla=int(time.time()))
print(f"Feedbacks guardados: {len(agente.historial)} | ε actual = {agente.epsilon:.3f}")
display(pd.DataFrame(agente.Q).round(3))
""")

md("""
### Paso 7.1: ronda de feedback — el usuario califica

En cada vuelta se toma un pedido **real** del dataset, la red predice, el agente escoge la acción (exploración o explotación), el LLM escribe la recomendación y **el usuario escribe su calificación de 1 a 5**. Se muestra cómo cambia Q.

Criterios para calificar: ¿es coherente con la predicción?, ¿las acciones son concretas y útiles para ese pedido?, ¿la estrategia tiene sentido (p. ej. no ofrecer cupón a un pedido rápido)?
""")
code("""
df = pd.read_csv(CSV_PATH)
pool = df[RAW_COLS].dropna().sample(800, random_state=7).reset_index(drop=True)
pred_pool = predictor.predecir(pool).reset_index(drop=True)
pred_pool["estado"] = pred_pool["resultado"].map(estado_de)
print(pred_pool["estado"].value_counts())

def tomar_pedido(ronda):
    # alternamos los estados para que el agente aprenda en los tres
    estado = ESTADOS[ronda % 3]
    i = np.random.choice(np.where(pred_pool["estado"] == estado)[0])
    return pool.iloc[i], pred_pool.iloc[i]
""")
code("""
RONDAS = 9   # recomendaciones a calificar

for ronda in range(RONDAS):
    fila, pred = tomar_pedido(ronda)
    estado = estado_de(pred["resultado"])
    accion, modo = agente.elegir_accion(estado)
    texto = generar_descripcion_llm(llm, fila, pred, accion)

    print("=" * 100)
    print(f"RONDA {ronda+1}/{RONDAS} | estado: {estado} | acción: {accion} ({modo}, ε={agente.epsilon:.3f})")
    print(f"Pedido: {fila['Category']}, tráfico {fila['Traffic'].strip()}, clima {fila['Weather']}, "
          f"{fila['Vehicle'].strip()}, {fila['Order_Time']}")
    print(f"Predicción: {pred['resultado']} | {pred['minutos_estimados']:.0f} min | P(lenta) {pred['prob_lenta']:.0%}")
    print("-" * 100); print(texto); print("-" * 100)

    nota = ""
    while nota not in {"1", "2", "3", "4", "5"}:
        nota = input("Califica la recomendación de 1 a 5: ").strip()
    comentario = input("Comentario (opcional, Enter para omitir): ").strip()
    q_antes = agente.Q[estado][accion]
    r, q_nuevo = agente.feedback_estrellas(estado, accion, int(nota), comentario)
    print(f"→ {'RECOMPENSA' if r > 0 else 'CASTIGO' if r < 0 else 'NEUTRO'} r = {r:+.2f} | "
          f"Q({estado}, {accion}): {q_antes:.3f} → {q_nuevo:.3f}")
""")

md("""
### Paso 7.2: ¿qué aprendió el agente?

- **Tabla Q** como mapa de calor (verde = recompensa esperada, rojo = castigo). La mejor acción de cada estado es la que el agente explota.
- **Recompensa por feedback** y su promedio acumulado.
- **ε por feedback**: cómo pasa de exploración a explotación.
""")
code("""
def graficar_agente(ag, titulo=""):
    hist = pd.DataFrame(ag.historial)
    Q = pd.DataFrame(ag.Q)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    im = axes[0].imshow(Q.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(Q.shape[1]), Q.columns); axes[0].set_yticks(range(Q.shape[0]), Q.index)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            axes[0].text(j, i, f"{Q.values[i, j]:.2f}\\nn={ag.N[Q.columns[j]][Q.index[i]]}", ha="center", va="center", fontsize=9)
    axes[0].set_title(f"Tabla Q {titulo}"); fig.colorbar(im, ax=axes[0])
    if len(hist):
        x = np.arange(1, len(hist) + 1)
        axes[1].bar(x, hist["recompensa"], color=np.where(hist["recompensa"] >= 0, "#2e8b57", "#c0392b"))
        axes[1].plot(x, hist["recompensa"].expanding().mean(), "k-", lw=2.5, label="promedio acumulado")
        axes[1].axhline(0, color="gray", lw=0.8); axes[1].set_ylim(-1.1, 1.1); axes[1].legend()
        axes[2].plot(x, hist["epsilon"], "o-")
    axes[1].set_title("Recompensa por feedback"); axes[1].set_xlabel("feedback")
    axes[2].set_title("ε (exploración → explotación)"); axes[2].set_xlabel("feedback"); axes[2].set_ylim(0, 0.55)
    plt.tight_layout(); plt.show()

graficar_agente(agente, "(feedback real)")
for e in ESTADOS:
    print(f"Mejor acción aprendida para '{e}': {agente.ranking(e)[0]}  →  {ACCIONES[agente.ranking(e)[0]]}")
""")

md("""
### Paso 7.3: verificación del algoritmo con un usuario simulado

Con pocas calificaciones reales no se alcanza a ver toda la convergencia, así que comprobamos el algoritmo con un **usuario simulado** (agente aparte, **no se guarda** ni se mezcla con el feedback real). El usuario simulado tiene preferencias ocultas con ruido: en pedidos **rápidos** prefiere el despacho estándar, en **intermedios** la ruta alternativa y en **lentos** la compensación o reprogramación. Comparamos el agente Q-learning contra escoger la acción **al azar**.
""")
code("""
preferencias = {   # estrellas promedio que daría el usuario simulado (ocultas para el agente)
    "Rápida":     {"despacho_estandar": 4.6, "priorizar_tienda": 3.6, "ruta_alternativa": 2.6,
                   "reasignar_repartidor": 2.2, "avisar_cliente": 3.2, "compensar_reprogramar": 1.6},
    "Intermedia": {"despacho_estandar": 2.8, "priorizar_tienda": 3.4, "ruta_alternativa": 4.5,
                   "reasignar_repartidor": 3.3, "avisar_cliente": 3.9, "compensar_reprogramar": 2.4},
    "Lenta":      {"despacho_estandar": 1.5, "priorizar_tienda": 2.8, "ruta_alternativa": 3.4,
                   "reasignar_repartidor": 3.5, "avisar_cliente": 3.9, "compensar_reprogramar": 4.7},
}
rng = np.random.default_rng(0)
def usuario_simulado(estado, accion):
    return int(np.clip(round(rng.normal(preferencias[estado][accion], 0.8)), 1, 5))

sim = AgenteQ(ruta=None, semilla=0)          # ruta=None: no se guarda
estrellas_rl, estrellas_azar = [], []
for t in range(300):
    estado = ESTADOS[t % 3]
    accion, _ = sim.elegir_accion(estado)
    n = usuario_simulado(estado, accion); sim.feedback_estrellas(estado, accion, n); estrellas_rl.append(n)
    estrellas_azar.append(usuario_simulado(estado, rng.choice(list(ACCIONES))))

plt.figure(figsize=(10, 4))
plt.plot(pd.Series(estrellas_rl).rolling(30, min_periods=1).mean(), label="agente Q-learning")
plt.plot(pd.Series(estrellas_azar).rolling(30, min_periods=1).mean(), label="acción al azar")
plt.title("Usuario simulado: estrellas promedio (ventana de 30 feedbacks)"); plt.xlabel("feedback"); plt.ylabel("★")
plt.ylim(1, 5); plt.legend(); plt.show()

print(f"Promedio de los últimos 100 feedbacks → Q-learning: {np.mean(estrellas_rl[-100:]):.2f} ★ | azar: {np.mean(estrellas_azar[-100:]):.2f} ★")
for e in ESTADOS:
    real = max(preferencias[e], key=preferencias[e].get)
    print(f"{e:<11} aprendió: {sim.ranking(e)[0]:<22} | preferida real: {real:<22} | {'✔' if sim.ranking(e)[0] == real else '✘'}")
graficar_agente(sim, "(usuario simulado, 300 feedbacks)")
""")

md("""
## Paso 8: recomendación con la política aprendida (explotación)

Para los tres pedidos del Paso 3, ahora la estrategia es la de **mayor Q** según el feedback real del usuario.
""")
code("""
for i in range(len(pedidos_demo)):
    fila, pred = pedidos_demo.iloc[i], pred_demo.iloc[i]
    estado = estado_de(pred["resultado"])
    accion = agente.ranking(estado)[0]
    texto = generar_descripcion_llm(llm, fila, pred, accion)
    print("=" * 100)
    print(f"PEDIDO {i+1}: {pred['resultado']} | {pred['minutos_estimados']:.0f} min | acción aprendida: {accion} "
          f"(Q={agente.Q[estado][accion]:.3f})")
    print("-" * 100); print(texto)
""")

md("""
## Paso 9: despliegue web — API FastAPI + `index.html`

- `api.py`: servicio web con los puntos de acceso `GET /` (página), `GET /salud`, `GET /opciones`, `POST /predict` (red → agente RL → LLM) y `POST /feedback` (calificación → actualización de la tabla Q), `GET /q_tabla`.
- `static/index.html`: formulario del pedido, predicción, recomendación del LLM, calificación con estrellas, 👍/👎 para las otras estrategias del agente y la tabla Q en vivo.

En Colab la API corre en segundo plano y se publica con un túnel de **Cloudflare** (link `https://….trycloudflare.com`, sin crear cuenta). Usa **el mismo agente** del notebook, así que el feedback de la web también entra en las gráficas del Paso 10.
""")
writefile("api.py")
writefile("static/index.html")
code("""
import threading, uvicorn
os.environ["API_AUTOLOAD"] = "0"          # la app se crea con los objetos ya cargados
import api; importlib.reload(api)
app = api.crear_app(predictor=predictor, llm=llm, agente=agente)

if "servidor" in globals():               # si se vuelve a correr la celda, se apaga el servidor anterior
    servidor.should_exit = True; time.sleep(3)
servidor = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
threading.Thread(target=servidor.run, daemon=True).start()
time.sleep(3)

import requests
print(requests.get("http://127.0.0.1:8000/salud").json())
""")
code("""
# Prueba de la API como lo haría la página: POST /predict y POST /feedback
pedido = pedidos_demo.iloc[2].to_dict()
r = requests.post("http://127.0.0.1:8000/predict", json=pedido).json()
print(json.dumps({k: v for k, v in r.items() if k != "recomendacion_llm"}, ensure_ascii=False, indent=1))
print("\\nRECOMENDACIÓN DEL LLM:\\n" + r["recomendacion_llm"])
""")
code("""
# Link público (túnel de Cloudflare). Abrirlo, llenar el formulario y calificar.
import subprocess, re
if not os.path.exists("cloudflared"):
    subprocess.run(["wget", "-q", "-O", "cloudflared",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    os.chmod("cloudflared", 0o755)
tunel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
URL = None
for _ in range(200):
    linea = tunel.stdout.readline()
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", linea)
    if m:
        URL = m.group(0); break
print("Página web pública:", URL)

# Alternativa sin túnel (solo funciona en este navegador):
from google.colab.output import eval_js
print("Link de Colab:", eval_js("google.colab.kernel.proxyPort(8000)"))
""")

md("""
## Paso 10: resultados finales del aprendizaje por refuerzo y descarga

Volvemos a graficar con **todo** el feedback (notebook + página web) y descargamos la app completa (`despliegue_llm_rl.zip`, con la tabla Q aprendida) y el registro de calificaciones.
""")
code("""
hist = pd.DataFrame(agente.historial)
hist.to_csv(f"{CARPETA}/feedback_usuarios.csv", index=False)
print(f"Feedbacks totales: {len(hist)}")
if len(hist):
    display(hist[["fecha", "estado", "accion", "tipo", "recompensa", "q_antes", "q_despues", "epsilon"]].tail(15).round(3))
    mitad = len(hist) // 2
    if mitad:
        print(f"Recompensa promedio — primera mitad: {hist['recompensa'][:mitad].mean():+.3f} | "
              f"segunda mitad: {hist['recompensa'][mitad:].mean():+.3f}")
graficar_agente(agente, "(final)")
""")
code("""
shutil.make_archive("despliegue_llm_rl", "zip", CARPETA)
files.download("despliegue_llm_rl.zip")
files.download(f"{CARPETA}/feedback_usuarios.csv")
""")

md("""
## Conclusiones

- **Modelo neuronal → LLM por pipeline:** la predicción de la red (categoría, minutos y probabilidad de entrega lenta) entra al prompt junto con los datos del pedido y la estrategia del agente; el LLM pre-entrenado **Qwen2.5-3B-Instruct** la convierte en recomendaciones concretas para la tienda, el repartidor y el cliente. El rol del sistema y la verificación de coherencia hacen que el texto respete la predicción y no invente datos.
- **Aprendizaje por refuerzo con feedback humano:** la calificación del usuario es la recompensa (o castigo) del **Q-learning**; la tabla Q de 3 estados × 6 acciones se actualiza con α y γ, y la política ε-greedy pasa de explorar a explotar. Con el usuario simulado el agente encontró la acción preferida en los tres estados y superó claramente a escoger al azar.
- **Despliegue:** API FastAPI (`/predict`, `/feedback`) + `index.html`; el mismo código corre local (`uvicorn api:app`) o en Colab con link público.
- **Límite:** el LLM no se re-entrena; lo que aprende el agente es **qué estrategia pedirle** al LLM para cada tipo de predicción, que es la forma práctica de aplicar RL con feedback sin GPU de entrenamiento.
""")

nb = {"cells": celdas, "metadata": {
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
destino = AQUI / "Proyecto_8_Corte3_LLM_RL.ipynb"
destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Escrito:", destino, "| celdas:", len(celdas))
