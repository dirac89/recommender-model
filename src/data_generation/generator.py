import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)

def generate_synthetic_data(n_samples=1000000, n_flights=1000, n_agencies=20):
    flights = [f"FLIGHT_{i}" for i in range(1, n_flights + 1)]
    agencies = [f"AGENCY_{i}" for i in range(1, n_agencies + 1)]
    routes = ['MAD-BCN', 'BCN-MAD', 'MAD-JFK', 'JFK-MAD', 'MAD-CDG', 'MAD-MEX', 'MAD-MIA', 'MAD-EZE', 'MAD-ORY', 'MAD-LAX']
    flight_classes = ['Y', 'J', 'F']

    rows = []

    for _ in range(n_samples):
        flight_id = random.choice(flights)
        agency_id = random.choice(agencies)
        route = random.choice(routes)
        haul = 'short' if 'MAD' in route and 'BCN' in route else 'long' if 'JFK' in route and 'MEX' in route and 'MIA' in route else 'medium'
        international = 1 if 'JFK' in route or 'CDG' in route or 'MEX' in route or 'MIA' in route or 'EZE' in route else 0
        flight_class = random.choice(flight_classes)

        # Usar un rango de fechas más amplio (desde hace 2 meses hasta 4 meses en el futuro)
        base_date = datetime(2026, 3, 1)
        departure_date = base_date + timedelta(days=random.randint(-60, 120))
        days_to_departure = (departure_date - base_date).days
        departure_hour = np.random.randint(0, 24)
        weekday = departure_date.strftime('%A')
        
        # Seasonality logic
        month = departure_date.month
        if month in [12, 1, 2]: season = 'winter'
        elif month in [3, 4, 5]: season = 'spring'
        elif month in [6, 7, 8]: season = 'summer'
        else: season = 'autumn'

        agency_rating = np.clip(np.random.normal(loc=0.7, scale=0.15), 0, 1)
        
        # Base demand influenced by seasonality and class
        base_demand_lam = 20
        if season == 'summer': base_demand_lam *= 1.4
        if flight_class == 'F': base_demand_lam *= 0.3
        elif flight_class == 'J': base_demand_lam *= 0.6
        
        past_sales = np.random.poisson(lam=base_demand_lam)
        past_materialization = np.clip(np.random.beta(a=2, b=1.5), 0, 1)
        
        # Realistic Target: past_sales * past_materialization * (1 + agency_rating) + noise
        # Added a non-linear interaction with price (proxy) and seasonality
        price_proxy = 1.0
        if season == 'summer': price_proxy = 1.5
        if haul == 'long': price_proxy *= 2.0
        
        # Demand elasticity: higher price -> lower recommended seats
        recommended_seats_raw = (past_sales * past_materialization * (1 + agency_rating)) / (price_proxy**0.2)
        noise = np.random.normal(0, 2)
        recommended_seats = int(np.clip(recommended_seats_raw + noise, 2, 80))

        initial_seats_assigned = int(np.clip(recommended_seats * np.random.uniform(0.8, 1.2), 5, 100))
        flight_capacity = np.random.randint(100, 300)

        # Outcome (only for historical data analysis, not for training features)
        seats_sold = min(np.random.binomial(n=initial_seats_assigned, p=past_materialization), initial_seats_assigned)
        
        load_factor_expected = np.clip(np.random.normal(loc=0.82, scale=0.05), 0.6, 0.95)
        load_factor_actual = seats_sold / flight_capacity
        
        # Lagged features from "previous" flights (simulated with noise)
        load_factor_lag_3 = np.clip(load_factor_expected + np.random.normal(0, 0.05), 0, 1)
        load_factor_lag_7 = np.clip(load_factor_expected + np.random.normal(0, 0.08), 0, 1)
        materialization_actual = seats_sold / initial_seats_assigned if initial_seats_assigned > 0 else 0

        rows.append({
            'flight_id': flight_id,
            'agency_id': agency_id,
            'departure_date': departure_date.date(),
            'days_to_departure': days_to_departure,
            'route': route,
            'haul': haul,
            'flight_class': flight_class,
            'weekday': weekday,
            'season': season,
            'departure_hour': departure_hour,
            'international': international,
            'flight_capacity': flight_capacity,
            'load_factor_expected': load_factor_expected,
            'load_factor_actual': load_factor_actual,
            'load_factor_lag_3': load_factor_lag_3,
            'load_factor_lag_7': load_factor_lag_7,
            'agency_rating': agency_rating,
            'past_sales': past_sales,
            'past_materialization': past_materialization,
            'initial_seats_assigned': initial_seats_assigned,
            'seats_sold': seats_sold,
            'materialization_actual': materialization_actual,
            'recommended_seats': recommended_seats
        })

    df = pd.DataFrame(rows)
    return df


def generate_synthetic_data_other(n_samples=1000000, n_flights=1000, n_agencies=20):
    flights = [f"FLIGHT_{i}" for i in range(1, n_flights + 1)]
    agencies = [f"AGENCY_{i}" for i in range(1, n_agencies + 1)]
    routes = ['MAD-BCN', 'BCN-MAD', 'MAD-JFK', 'JFK-MAD', 'MAD-CDG', 'MAD-MEX', 'MAD-MIA', 'MAD-EZE', 'MAD-ORY', 'MAD-LAX']
    flight_classes = ['Y', 'J', 'F']


    rows = []

    for _ in range(n_samples):
        flight_id = random.choice(flights)
        agency_id = random.choice(agencies)
        route = random.choice(routes)
        haul = 'short' if 'MAD' in route and 'BCN' in route else 'long' if 'JFK' in route and 'MEX' in route and 'MIA' in route else 'medium'
        flight_class = random.choice(flight_classes)

        departure_date = datetime.today() + timedelta(days=random.randint(1, 180))
        days_to_departure = (departure_date - datetime.today()).days

        agency_rating = np.clip(np.random.normal(loc=0.7, scale=0.15), 0, 1)
        revenue = 1000 * np.random.poisson(lam=20)


        rows.append({
            'flight_id': flight_id,
            'agency_id': agency_id,
            'departure_date': departure_date.date(),
            'days_to_departure': days_to_departure,
            'route': route,
            'haul': haul,
            'flight_class': flight_class,
            'agency_rating': agency_rating,
            'revenue': revenue
        })

    df = pd.DataFrame(rows)
    return df



def generate_inference_data(flight_id="FLIGHT_PRED_001", n_agencies=20):
    """
    Genera datos para un solo vuelo en el futuro para probar la inferencia.
    """
    agencies = [f"AGENCY_{i}" for i in range(1, n_agencies + 1)]
    routes = ['MAD-BCN', 'BCN-MAD', 'MAD-JFK', 'JFK-MAD', 'MAD-CDG', 'MAD-MEX', 'MAD-MIA', 'MAD-EZE', 'MAD-ORY', 'MAD-LAX']
    flight_classes = ['Y', 'J', 'F']
    
    route = random.choice(routes)
    haul = 'short' if 'MAD' in route and 'BCN' in route else 'long' if 'JFK' in route and 'MEX' in route and 'MIA' in route else 'medium'
    international = 1 if 'JFK' in route or 'CDG' in route or 'MEX' in route or 'MIA' in route or 'EZE' in route else 0
    
    # Fecha futura: hoy + 30 días
    departure_date = datetime.now() + timedelta(days=30)
    departure_hour = np.random.randint(0, 24)
    weekday = departure_date.strftime('%A')
    
    month = departure_date.month
    if month in [12, 1, 2]: season = 'winter'
    elif month in [3, 4, 5]: season = 'spring'
    elif month in [6, 7, 8]: season = 'summer'
    else: season = 'autumn'

    rows = []
    flight_capacity = np.random.randint(100, 300)

    for agency_id in agencies:
        flight_class = random.choice(flight_classes)
        agency_rating = np.clip(np.random.normal(loc=0.7, scale=0.15), 0, 1)
        
        # Simular ventas pasadas moderadas
        past_sales = np.random.poisson(lam=15)
        past_materialization = np.clip(np.random.beta(a=2, b=1.5), 0, 1)
        
        # Load factors ficticios pero coherentes
        load_factor_expected = np.clip(np.random.normal(loc=0.8, scale=0.05), 0.6, 0.95)
        
        rows.append({
            'flight_id': flight_id,
            'agency_id': agency_id,
            'departure_date': departure_date.date(),
            'days_to_departure': 30,
            'route': route,
            'haul': haul,
            'flight_class': flight_class,
            'weekday': weekday,
            'season': season,
            'departure_hour': departure_hour,
            'international': international,
            'flight_capacity': flight_capacity,
            'load_factor_expected': load_factor_expected,
            'load_factor_lag_3': np.clip(load_factor_expected + 0.02, 0, 1),
            'load_factor_lag_7': np.clip(load_factor_expected - 0.01, 0, 1),
            'agency_rating': agency_rating,
            'past_sales': past_sales,
            'past_materialization': past_materialization,
            'initial_seats_assigned': int(past_sales * 1.1)  # Valor proxy para inferencia
        })

    return pd.DataFrame(rows)


from src.common.config import RAW_DATA_PATH

if __name__ == "__main__":
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Dataset generado en {RAW_DATA_PATH}")
