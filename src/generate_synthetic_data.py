import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)

def generate_synthetic_data(n_samples=100000, n_flights=100, n_agencies=10):
    flights = [f"FLIGHT_{i}" for i in range(1, n_flights + 1)]
    agencies = [f"AGENCY_{i}" for i in range(1, n_agencies + 1)]
    routes = ['MAD-BCN', 'BCN-MAD', 'MAD-JFK', 'JFK-MAD', 'MAD-CDG', 'MAD-MEX', 'MAD-MIA', 'MAD-EZE', 'MAD-ORY', 'MAD-LAX']
    hauls = ['short', 'medium', 'long']
    flight_classes = ['Y', 'J', 'F']
    seasons = ['winter', 'spring', 'summer', 'autumn']

    rows = []

    for _ in range(n_samples):
        flight_id = random.choice(flights)
        agency_id = random.choice(agencies)
        route = random.choice(routes)
        haul = 'short' if 'MAD' in route and 'BCN' in route else 'long' if 'JFK' in route and 'MEX' in route and 'MIA' in route else 'medium'
        international = 1 if 'JFK' in route or 'CDG' in route or 'MEX' in route or 'MIA' in route or 'EZE' in route else 0
        flight_class = random.choice(flight_classes)

        departure_date = datetime.today() + timedelta(days=random.randint(1, 180))
        days_to_departure = (departure_date - datetime.today()).days
        departure_hour = np.random.randint(0, 24)
        weekday = departure_date.strftime('%A')
        season = random.choice(seasons)

        agency_rating = np.clip(np.random.normal(loc=0.7, scale=0.15), 0, 1)
        past_sales = np.random.poisson(lam=20)
        past_materialization = np.clip(np.random.beta(a=2, b=1.5), 0, 1)
        initial_seats_assigned = np.random.randint(5, 50)
        flight_capacity = np.random.randint(100, 220)

        seats_sold = min(np.random.binomial(n=initial_seats_assigned, p=past_materialization), initial_seats_assigned)
        recommended_seats = int(np.clip(past_sales * past_materialization * (1 + agency_rating), 5, 70))

        load_factor_expected = np.clip(np.random.normal(loc=0.82, scale=0.05), 0.6, 0.95)
        load_factor_actual = seats_sold / flight_capacity
        load_factor_lag_3 = np.clip(load_factor_actual + np.random.normal(0, 0.03), 0, 1)
        load_factor_lag_7 = np.clip(load_factor_actual + np.random.normal(0, 0.05), 0, 1)
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

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv("data/raw/synthetic_recommendation_dataset.csv", index=False)
    print("✅ Dataset generado en data/raw/synthetic_recommendation_dataset.csv")
