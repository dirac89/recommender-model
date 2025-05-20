import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from src.config import RAW_DATA_PATH, PROCESSED_DIR, TARGET_COL, CATEGORICAL_COLUMNS, TEST_SIZE, RANDOM_STATE

class Preprocessor:
    def __init__(self, input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR, target_col=TARGET_COL):
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_col = target_col
        self.categorical_columns = CATEGORICAL_COLUMNS
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.input_path, parse_dates=['departure_date'])

    def _extract_season(self, date: pd.Timestamp) -> str:
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    def _extract_day_type(self, date: pd.Timestamp) -> str:
        return 'weekend' if date.weekday() >= 5 else 'working_day'

    def enrich_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['season_extracted'] = df['departure_date'].apply(self._extract_season)
        df['day_type'] = df['departure_date'].apply(self._extract_day_type)
        return df

    def cluster_routes(self, df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
        # Calculamos métricas por ruta
        df['sales_ratio'] = df['seats_sold'] / df['flight_capacity']
        
        route_stats = df.groupby('route').agg(
            mean_materialization=('materialization_actual', 'mean'),
            mean_sales_ratio=('sales_ratio', 'mean')
        ).fillna(0)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        route_stats['route_cluster'] = kmeans.fit_predict(route_stats[['mean_materialization', 'mean_sales_ratio']])

        df = df.merge(route_stats[['route_cluster']], left_on='route', right_index=True, how='left')
        df['route_cluster'] = df['route_cluster'].astype(str)

        # Limpieza si se desea quitar la columna auxiliar
        df.drop(columns='sales_ratio', inplace=True)

        return df


    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna({
            'past_sales': 0,
            'past_materialization': df['past_materialization'].mean(),
            'agency_rating': 0.5,
            'load_factor_expected': df['load_factor_expected'].mean(),
            'load_factor_actual': df['load_factor_actual'].mean(),
            'load_factor_lag_3': df['load_factor_lag_3'].mean(),
            'load_factor_lag_7': df['load_factor_lag_7'].mean(),
            'materialization_actual': df['materialization_actual'].mean(),
        })

        df = self.enrich_datetime_features(df)
        df = self.cluster_routes(df)

        for col in self.categorical_columns + ['season_extracted', 'day_type', 'route_cluster']:
            df[col] = df[col].astype(str)

        return df

    def split_and_save(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        X_train.to_csv(f"{self.output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{self.output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{self.output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{self.output_dir}/y_test.csv", index=False)

        print("✅ Datos procesados y guardados en:", self.output_dir)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    df = preprocessor.load_data()
    df = preprocessor.preprocess(df)
    preprocessor.split_and_save(df)
