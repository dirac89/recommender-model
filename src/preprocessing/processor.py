import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from src.common.config import RAW_DATA_PATH, PROCESSED_DIR, TARGET_COL, CATEGORICAL_COLUMNS, TEST_SIZE, RANDOM_STATE, CUTOFF_DATE

class Preprocessor:
    def __init__(self, input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR, target_col=TARGET_COL):
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_col = target_col
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.leaky_features = ['materialization_actual', 'load_factor_actual', 'seats_sold']
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.input_path, parse_dates=['departure_date'])

    def _extract_season(self, date: pd.Timestamp) -> str:
        month = date.month
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: season = 'summer'
        else: return 'autumn'

    def _extract_day_type(self, date: pd.Timestamp) -> str:
        return 'weekend' if date.weekday() >= 5 else 'working_day'

    def enrich_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['season_extracted'] = df['departure_date'].apply(self._extract_season)
        df['day_type'] = df['departure_date'].apply(self._extract_day_type)
        return df

    def cluster_routes(self, df_train: pd.DataFrame, df_test: pd.DataFrame, n_clusters=4) -> tuple:
        # Calculamos métricas por ruta usando SOLO datos de entrenamiento para evitar leakage
        df_train['sales_ratio'] = df_train['past_sales'] / df_train['flight_capacity']
        
        route_stats = df_train.groupby('route').agg(
            mean_materialization=('past_materialization', 'mean'),
            mean_sales_ratio=('sales_ratio', 'mean')
        ).fillna(0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        route_stats['route_cluster'] = kmeans.fit_predict(route_stats[['mean_materialization', 'mean_sales_ratio']])

        # Aplicar el clustering a ambos datasets
        df_train = df_train.merge(route_stats[['route_cluster']], left_on='route', right_index=True, how='left')
        df_test = df_test.merge(route_stats[['route_cluster']], left_on='route', right_index=True, how='left')
        
        df_train['route_cluster'] = df_train['route_cluster'].fillna('-1').astype(str)
        df_test['route_cluster'] = df_test['route_cluster'].fillna('-1').astype(str)

        df_train.drop(columns='sales_ratio', inplace=True)
        return df_train, df_test


    def preprocess_and_split(self, df: pd.DataFrame):
        # 1. Limpieza básica
        df = df.fillna({
            'past_sales': 0,
            'past_materialization': df['past_materialization'].mean(),
            'agency_rating': 0.5,
            'load_factor_expected': df['load_factor_expected'].mean(),
            'load_factor_lag_3': df['load_factor_lag_3'].mean(),
            'load_factor_lag_7': df['load_factor_lag_7'].mean(),
        })

        # 2. Enriquecer fechas
        df = self.enrich_datetime_features(df)

        # 3. Split Temporal
        cutoff = pd.to_datetime(CUTOFF_DATE)
        train_mask = df['departure_date'] < cutoff
        df_train = df[train_mask].copy()
        df_test = df[~train_mask].copy()
        
        print(f"📅 Split temporal aplicado: {CUTOFF_DATE}")
        print(f"   Train: {len(df_train)} filas (desde {df_train['departure_date'].min().date()} hasta {df_train['departure_date'].max().date()})")
        print(f"   Test:  {len(df_test)} filas (desde {df_test['departure_date'].min().date()} hasta {df_test['departure_date'].max().date()})")

        # 4. Clustering (usando solo train stats)
        df_train, df_test = self.cluster_routes(df_train, df_test)

        # 5. Convertir categóricas
        cat_cols = self.categorical_columns + ['season_extracted', 'day_type', 'route_cluster']
        for col in cat_cols:
            if col in df_train.columns:
                df_train[col] = df_train[col].astype(str)
                df_test[col] = df_test[col].astype(str)

        # 6. Eliminar leaky features del set de entrenamiento
        # Pero las mantenemos en y_test / y_train (el target es recommended_seats, que NO es leaky)
        X_train = df_train.drop(columns=[self.target_col] + self.leaky_features)
        X_test = df_test.drop(columns=[self.target_col] + self.leaky_features)
        
        y_train = df_train[self.target_col]
        y_test = df_test[self.target_col]

        # Guardar
        X_train.to_csv(f"{self.output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{self.output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{self.output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{self.output_dir}/y_test.csv", index=False)

        print("✅ Datos procesados y guardados en:", self.output_dir)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    df = preprocessor.load_data()
    preprocessor.preprocess_and_split(df)
