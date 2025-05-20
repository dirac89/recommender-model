# === CONFIGURACIÓN ===

PROJECT_NAME = recommender-catboost
PYTHON = poetry run python

# === COMANDOS DE PYTHON ===

pipeline:
	@echo "🚀 Ejecutando pipeline completo en local..."
	$(PYTHON) src/run_pipeline.py

generate:
	@echo "🧪 Generando datos sintéticos..."
	$(PYTHON) src/generate_synthetic_data.py

preprocess:
	@echo "⚙️ Preprocesando datos..."
	$(PYTHON) src/preprocessing.py

train:
	@echo "🎯 Entrenando modelo..."
	$(PYTHON) src/train.py

# === DOCKER ===

docker-build:
	@echo "🐳 Construyendo imagen Docker..."
	docker-compose build --no-cache

docker-up:
	@echo "🔧 Ejecutando contenedor con pipeline completo..."
	docker-compose up --build

docker-shell:
	@echo "🛠️ Ingresando al contenedor de Docker..."
	docker-compose run $(PROJECT_NAME)

docker-clean:
	@echo "🧹 Deteniendo y eliminando contenedor..."
	docker-compose down

# === UTILIDADES ===

check-version:
	@echo "🐍 Verificando versión de scikit-learn en entorno local..."
	$(PYTHON) -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

.PHONY: pipeline generate preprocess train docker-build docker-up docker-shell docker-clean check-version