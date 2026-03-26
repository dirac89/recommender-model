# === CONFIGURACIÓN ===
PYTHON = venv/bin/python

# === COMANDOS PRINCIPALES ===

help:
	@echo "✈️  Flight Recommender - Makefile"
	@echo "--------------------------------"
	@echo "make pipeline      : Ejecuta el flujo completo (Data -> Prep -> Train -> Demo)"
	@echo "make api           : Inicia el servidor de producción (FastAPI)"
	@echo "make monitor       : Ejecuta el análisis de Population Stability Index (PSI)"
	@echo "make test-api      : Envía una petición de ejemplo a la API local"
	@echo "make clean         : Elimina archivos temporales y modelos"

pipeline:
	@echo "🚀 Ejecutando pipeline completo (Nivel 2.2)..."
	$(PYTHON) src/run_pipeline.py

predict:
	@echo "🎲 Generando datos de inferencia y guardando predicciones..."
	$(PYTHON) src/run_inference.py

api:
	@echo "🌐 Iniciando API de Producción (Puerto 8000)..."
	$(PYTHON) src/inference/api.py

monitor:
	@echo "📡 Analizando Drift de datos (PSI)..."
	$(PYTHON) src/common/monitoring.py

mlflow-ui:
	@echo "📊 Iniciando Dashboard de MLflow (Puerto 5000)..."
	$(PYTHON) -m mlflow ui --port 5000 --host 127.0.0.1

test-api:
	@echo "🧪 Enviando petición de prueba a la API..."
	curl -X POST "http://localhost:8000/predict/single" \
		-H "Content-Type: application/json" \
		-d '{"agency_id": "AGENCY_3", "days_to_departure": 45, "route": "MAD-MEX", "haul": "medium", "flight_class": "F", "weekday": "Wednesday", "season": "spring", "departure_hour": 21, "international": 1, "flight_capacity": 207, "load_factor_expected": 0.81, "load_factor_lag_3": 0.78, "load_factor_lag_7": 0.82, "agency_rating": 0.56, "past_sales": 1.0, "past_materialization": 0.04, "initial_seats_assigned": 5.0, "season_extracted": "spring", "day_type": "working_day", "route_cluster": "3"}'

clean:
	@echo "🧹 Limpiando el entorno..."
	rm -rf models/*.cbm
	rm -rf data/processed/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: help pipeline api monitor test-api clean