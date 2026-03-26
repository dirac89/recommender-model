import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.generator import generate_synthetic_data
from src.preprocessing.processor import Preprocessor
from src.training.trainer import RecommenderTrainer
from src.common.config import RAW_DATA_PATH

def main():
    print("🚀 Generando datos sintéticos...")
    df = generate_synthetic_data()
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print("✅ Datos generados en:", RAW_DATA_PATH)

    print("⚙️  Preprocesando datos...")
    preprocessor = Preprocessor()
    df = preprocessor.load_data()
    preprocessor.preprocess_and_split(df)

    print("🎯 Entrenando modelo...")
    trainer = RecommenderTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    model = trainer.train(X_train, y_train)
    
    print("📈 Evaluando modelo...")
    trainer.evaluate_model(model, X_test, y_test)
    
    print("💾 Guardando modelo...")
    trainer.save_model(model)

    print("🏁 Pipeline finalizado exitosamente.")

if __name__ == "__main__":
    main()
