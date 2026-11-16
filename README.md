# Proyecto 8 — Corte 3 Final: red neuronal + LLM + aprendizaje por refuerzo

Predicción del tiempo de entrega de pedidos de Amazon (Rápida ≤ 70 min · Intermedia · Lenta ≥ 180 min) con una red neuronal, recomendaciones escritas por un LLM (Qwen2.5-3B-Instruct) y un agente de Q-learning que aprende de las calificaciones del usuario.

**Integrantes:** Kevin Tapias, Sebastian Villera y Anuar Marriaga.

## Qué hay aquí
| Archivo | Para qué |
|---|---|
| `Proyecto_8_Corte3_LLM_RL.ipynb` | Notebook para **Google Colab (GPU T4)**. Es el que se ejecuta y se entrega con sus salidas. |
| `despliegue_llm_rl/` | La aplicación web final (la misma que escribe el notebook). |
| `despliegue_llm_rl/recomendador.py` | LLM (Qwen2.5 por `pipeline` de transformers) + diseño del prompt. |
| `despliegue_llm_rl/rl_feedback.py` | Agente Q-learning: estados, acciones, tabla Q, recompensas, ε-greedy. |
| `despliegue_llm_rl/api.py` | API FastAPI: `/predict`, `/feedback`, `/q_tabla`, `/salud`, `/opciones`. |
| `despliegue_llm_rl/static/index.html` | Página: formulario, predicción, recomendación, estrellas y 👍/👎, tabla Q. |
| `construir_notebook.py` | Regenera el notebook desde los archivos de `despliegue_llm_rl/` (solo si se cambian). |
| `despliegue_llm_rl/q_tabla.json`, `feedback_usuarios.csv` | Tabla Q aprendida y calificaciones de la ejecución final en Colab. |
| `vercel.json` | Publica la página (`despliegue_llm_rl/static`) en Vercel. |

## Cómo ejecutar en Colab (recomendado)
1. Abrir el notebook en Colab → Entorno de ejecución → Cambiar tipo → **GPU T4**.
2. Correr celda por celda. En el Paso 2 subir `despliegue (1).zip` y `amazon_delivery.csv` (modelo del Corte 2 y dataset de Amazon).
3. En el Paso 7.1 calificar las 9 recomendaciones escribiendo 1–5.
4. En el Paso 9 abrir el link `https://….trycloudflare.com`, probar los 3 ejemplos y calificar (capturas para el informe).
5. Paso 10: gráficas finales y descarga. Luego **Archivo → Descargar → .ipynb** para entregar el notebook ejecutado.

## Cómo ejecutar local (PC sin GPU)
```
cd despliegue_llm_rl
pip install -r requirements.txt
set MODELO_LLM=Qwen/Qwen2.5-0.5B-Instruct      (en CPU el de 3B es muy lento)
uvicorn api:app --port 8000
```
Abrir http://127.0.0.1:8000

## Página en Vercel
Vercel solo aloja la página (no tiene GPU para Qwen ni la red). La API corre en Colab:
1. Correr el notebook hasta el Paso 9 y copiar el link `https://….trycloudflare.com`.
2. Abrir la página de Vercel, desplegar **Servidor de la API**, pegar el link y pulsar **Conectar**
   (o abrir `https://<página-de-vercel>/?api=https://….trycloudflare.com`).
