import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate_synthetic_data import generate_synthetic_data
from src.preprocessing import Preprocessor
from src.train import RecommenderTrainer
from src.config import RAW_DATA_PATH

def main():
    print("🚀 Generando datos sintéticos...")
    df = generate_synthetic_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    print("✅ Datos generados en:", RAW_DATA_PATH)

    print("⚙️  Preprocesando datos...")
    preprocessor = Preprocessor()
    df = preprocessor.preprocess(df)
    preprocessor.split_and_save(df)

    print("🎯 Entrenando modelo...")
    trainer = RecommenderTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    model = trainer.train(X_train, y_train)
    trainer.save_model(model)

    model = trainer.train(X_train, y_train)
    trainer.save_model(model)

    print("📈 Evaluando modelo...")
    trainer.evaluate_model(model, X_test, y_test)

    print("🏁 Pipeline finalizado exitosamente.")

if __name__ == "__main__":
    main()
