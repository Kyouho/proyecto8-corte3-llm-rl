"""Aprendizaje por refuerzo (Q-learning) para calificar las recomendaciones del LLM.

- Estados (s): la categoría que predijo la red neuronal -> Rápida, Intermedia, Lenta.
- Acciones (a): estrategias de recomendación predeterminadas (ACCIONES). El agente escoge una
  y el LLM la desarrolla en un texto adaptado al pedido.
- Recompensa (r): el feedback del usuario. Estrellas 1..5 -> r = (estrellas - 3) / 2
  (1★ = -1 castigo fuerte, 3★ = 0 neutro, 5★ = +1 premio). Me gusta = +1, No me gusta = -1.
- Tabla Q: Q[s][a] = qué tan buena es la acción a en el estado s, según lo aprendido.
- Actualización (Q-learning):  Q(s,a) <- Q(s,a) + alfa * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
  Cada pedido es un episodio de un solo paso (después de recomendar no hay estado siguiente),
  así que s' es terminal y el objetivo queda r. gamma se deja como hiperparámetro por si se
  encadenan pasos (p. ej. seguimiento del mismo pedido).
- Política epsilon-greedy: con probabilidad epsilon explora (acción al azar) y si no, explota
  (acción con mayor Q). epsilon baja con cada feedback: exploración -> explotación.
"""
import json
import os
from datetime import datetime

import numpy as np

ESTADOS = ["Rápida", "Intermedia", "Lenta"]

ACCIONES = {
    "despacho_estandar": "Mantener el despacho estándar y confirmar al cliente la hora estimada de llegada",
    "priorizar_tienda": "Priorizar la preparación y el despacho inmediato del pedido en la tienda",
    "ruta_alternativa": "Planear una ruta alternativa que evite las vías congestionadas y avisar al repartidor",
    "reasignar_repartidor": "Reasignar el pedido a un repartidor o vehículo más adecuado para las condiciones",
    "avisar_cliente": "Avisar al cliente de forma proactiva con una nueva hora estimada y seguimiento en vivo",
    "compensar_reprogramar": "Ofrecer una compensación (cupón o envío gratis) o reprogramar la franja de entrega",
}

# Valores iniciales predeterminados (conocimiento previo del negocio, en la escala de la recompensa).
# Son pequeños para que el feedback del usuario los cambie rápido.
Q_INICIAL = {
    "Rápida":     {"despacho_estandar": 0.30, "priorizar_tienda": 0.10, "ruta_alternativa": 0.00,
                   "reasignar_repartidor": -0.10, "avisar_cliente": 0.10, "compensar_reprogramar": -0.20},
    "Intermedia": {"despacho_estandar": 0.05, "priorizar_tienda": 0.15, "ruta_alternativa": 0.15,
                   "reasignar_repartidor": 0.05, "avisar_cliente": 0.20, "compensar_reprogramar": -0.05},
    "Lenta":      {"despacho_estandar": -0.20, "priorizar_tienda": 0.10, "ruta_alternativa": 0.15,
                   "reasignar_repartidor": 0.10, "avisar_cliente": 0.25, "compensar_reprogramar": 0.20},
}


def estrellas_a_recompensa(estrellas):
    return (float(estrellas) - 3.0) / 2.0


class AgenteQ:
    def __init__(self, ruta="q_tabla.json", alfa=0.3, gamma=0.9,
                 epsilon_inicial=0.5, epsilon_min=0.05, decaimiento=0.95, semilla=None):
        self.ruta = ruta
        self.alfa, self.gamma = alfa, gamma
        self.epsilon_inicial, self.epsilon_min, self.decaimiento = epsilon_inicial, epsilon_min, decaimiento
        self.rng = np.random.default_rng(semilla)
        self.Q = {s: dict(Q_INICIAL[s]) for s in ESTADOS}
        self.N = {s: {a: 0 for a in ACCIONES} for s in ESTADOS}
        self.historial = []
        if ruta and os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
            self.Q, self.N, self.historial = datos["Q"], datos["N"], datos["historial"]

    # ---------------- política ----------------
    @property
    def epsilon(self):
        return max(self.epsilon_min, self.epsilon_inicial * self.decaimiento ** len(self.historial))

    def ranking(self, estado):
        """Acciones ordenadas de mayor a menor Q (lo que el agente cree mejor)."""
        return sorted(self.Q[estado], key=self.Q[estado].get, reverse=True)

    def elegir_accion(self, estado):
        """epsilon-greedy. Devuelve (acción, 'exploración' | 'explotación')."""
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(list(ACCIONES))), "exploración"
        return self.ranking(estado)[0], "explotación"

    # ---------------- aprendizaje ----------------
    def actualizar(self, estado, accion, recompensa, estado_siguiente=None, info=None):
        q_actual = self.Q[estado][accion]
        futuro = 0.0 if estado_siguiente is None else max(self.Q[estado_siguiente].values())
        objetivo = recompensa + self.gamma * futuro
        self.Q[estado][accion] = q_actual + self.alfa * (objetivo - q_actual)
        self.N[estado][accion] += 1
        self.historial.append({"fecha": datetime.now().isoformat(timespec="seconds"),
                               "estado": estado, "accion": accion, "recompensa": float(recompensa),
                               "q_antes": q_actual, "q_despues": self.Q[estado][accion],
                               "epsilon": self.epsilon, **(info or {})})
        self.guardar()
        return self.Q[estado][accion]

    def feedback_estrellas(self, estado, accion, estrellas, comentario=""):
        if not 1 <= int(estrellas) <= 5:
            raise ValueError("Las estrellas deben estar entre 1 y 5")
        r = estrellas_a_recompensa(estrellas)
        return r, self.actualizar(estado, accion, r, info={"tipo": "estrellas", "estrellas": int(estrellas),
                                                           "comentario": comentario})

    def feedback_me_gusta(self, estado, accion, me_gusta):
        r = 1.0 if me_gusta else -1.0
        return r, self.actualizar(estado, accion, r, info={"tipo": "me_gusta" if me_gusta else "no_me_gusta"})

    # ---------------- persistencia ----------------
    def guardar(self):
        if not self.ruta:
            return
        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump({"Q": self.Q, "N": self.N, "historial": self.historial}, f, ensure_ascii=False, indent=2)

    def tabla(self):
        """Tabla Q como lista de filas (para mostrar en pandas o en la web)."""
        return [{"estado": s, "accion": a, "Q": round(self.Q[s][a], 4), "veces": self.N[s][a]}
                for s in ESTADOS for a in ACCIONES]
