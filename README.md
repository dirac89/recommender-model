# ✈️ Flight Seat Recommender with CatBoost

Este proyecto implementa un sistema de recomendación que predice cuántas plazas deben ser cedidas por una aerolínea a agencias externas para comercializar un vuelo, utilizando técnicas de machine learning (CatBoost), un pipeline modular en Python y un entorno reproducible con Docker y Poetry.

---

## 🚀 Características

- 🔍 Preprocesamiento automatizado con extracción de features temporales
- 🧠 Modelo de regresión basado en **CatBoostRegressor**
- 🎯 Optimización de métricas (RMSE, MAE, R²)
- 🧪 Datos sintéticos generados programáticamente
- 🐳 Entorno reproducible con Docker + Docker Compose
- 📦 Gestión de dependencias con **Poetry**
- 🔁 Pipeline completo orquestado desde `run_pipeline.py`
- 🛠️ Comandos simplificados con `Makefile`

---

## 🗂️ Estructura del proyecto

```
ml_recommender_project/
├── data/
│   ├── raw/                 # Datos sintéticos generados
│   └── processed/           # Datos preprocesados (train/test)
├── models/                  # Modelo CatBoost entrenado
├── outputs/                 # Métricas de evaluación (JSON)
├── src/
│   ├── config.py            # Rutas y configuración global
│   ├── generate_synthetic_data.py
│   ├── preprocessing.py     # Limpieza y enriquecimiento
│   ├── train.py             # Entrenamiento y evaluación
│   └── run_pipeline.py      # Pipeline completo
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- [Docker](https://www.docker.com/) + [Docker Compose](https://docs.docker.com/compose/)

---

## 💻 Instalación local

```bash
# Clona el repositorio
git clone https://github.com/TU_USUARIO/catboost-recommender.git
cd catboost-recommender

# Instala dependencias
poetry install

# Ejecuta el pipeline completo
make pipeline
```

---

## 🐳 Usar con Docker

```bash
# Construir imagen limpia
make docker-build

# Ejecutar pipeline completo en contenedor
make docker-up

# Entrar al contenedor (modo interactivo)
make docker-shell

# Detener y eliminar contenedor
make docker-clean
```

---

## 📈 Evaluación

Las métricas de evaluación del modelo se guardan automáticamente en:

```bash
outputs/metrics.json
```

Incluyen:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

---

## 📬 Contacto

Desarrollado por Dirac89
✉️ [aguilerajavier58@gmail.com]  
🔗 [linkedin.com/in/javier-aguilera](https://www.linkedin.com/in/javier-aguilera-fernández/)

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.