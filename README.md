# ✈️ Flight Seat Recommender (Revenue Management System)

Este proyecto implementa un sistema de **Revenue Management** de grado producción para la asignación óptima de asientos de aerolíneas a agencias externas. Ha evolucionado de un prototipo básico a una infraestructura robusta de ML con servido en tiempo real y observabilidad.

---

## 🚀 Características (Niveles 1, 2 y 3)

### 📊 Modelado Predictivo e Incertidumbre
- **CatBoost + Pérdida Asimétrica (Quantile Loss)**: Penaliza la sobre-estimación de demanda para proteger el inventario.
- **Uncertainty Quantification (Conformal Prediction)**: Integración con **MAPIE** para proporcionar intervalos de confianza del 90% estadísticamente garantizados (**Coverage ~89%**).
- **Capa de Optimización Lineal**: Solver basado en `scipy.optimize.linprog` que maximiza la asignación de asientos respetando límites de capacidad y prioridades de agencia.

### 🌐 Producción y Servido (Nivel 3)
- **API de Inferencia (FastAPI)**: Endpoints rápidos para predicciones individuales (`/predict/single`) y optimización batch por vuelo (`/predict/batch`).
- **Observabilidad (PSI)**: Monitor de **Data Drift** (Population Stability Index) para detectar cambios en la distribución de features en el tráfico real.
- **Tracking Profesional (MLflow)**: Registro automático de experimentos, hiperparámetros y modelos serializados.

---

## 🗂️ Estructura del proyecto

```text
recommender-model/
├── src/
│   ├── common/           # Configuración, Monitorización (PSI) y Config de negocio
│   ├── data_generation/  # Generación de datos sintéticos estocásticos
│   ├── preprocessing/    # Pipeline de limpieza, codificación y clustering
│   ├── training/         # Trainer (MLflow + Conformal Prediction)
│   ├── inference/        # API (FastAPI) y Optimizer (Solver Lineal)
│   └── run_pipeline.py   # Orquestador del flujo completo
├── data/
│   ├── raw/ & processed/ # Almacenamiento de datasets
├── models/               # Modelos serializados y metadatos
├── mlruns/               # Logs y artefactos de MLflow
└── Makefile              # Comandos de ejecución rápida
```

---

## 💻 Ejecución (Makefile)

He simplificado la operación del sistema mediante un `Makefile`:

- **`make pipeline`**: Ejecuta el flujo completo (Data -> Train -> Eval -> Demo).
- **`make api`**: Inicia el servidor de producción en el puerto 8000.
- **`make monitor`**: Ejecuta el análisis de PSI para detectar drift.
- **`make mlflow-ui`**: Inicia el dashboard de MLflow (Puerto 5000).
- **`make predict`**: Genera datos de prueba futuros y guarda resultados en `data/inference/`.
- **`make test-api`**: Envía una petición de prueba a la API local.

---

## 📈 Evaluación y Métricas

- **R2 Score**: ~0.98 (Alta capacidad predictiva).
- **Coverage_90**: **89.08%** (Validación de intervalos conformales).
- **Total Assigned**: Verificación de que el solver consume el inventario de forma óptima bajo restricciones de capacidad.

---

## 📬 Contacto

Desarrollado por **Antigravity AI** en colaboración con **Dirac89**
✉️ [aguilerajavier58@gmail.com]  
🔗 [linkedin.com/in/javier-aguilera-fernanderz](https://www.linkedin.com/in/javier-aguilera-fernández/)

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.