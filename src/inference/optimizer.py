import numpy as np
from scipy.optimize import linprog
import pandas as pd

class SeatOptimizer:
    """
    Optimiza la distribución de asientos entre agencias dada una capacidad total.
    """
    def __init__(self, capacity_limit):
        self.capacity_limit = capacity_limit

    def optimize_allocation(self, agency_ids, predicted_demands, lower_bounds, upper_bounds, priorities=None):
        """
        Calcula la asignación óptima usando programación lineal.
        
        Args:
            agency_ids: List of agency identifiers
            predicted_demands: ML point predictions
            lower_bounds: Confidence interval lower bounds (CP)
            upper_bounds: Confidence interval upper bounds (CP)
            priorities: Optional weights (e.g., agency_rating) to prioritize certain agencies
        
        Returns:
            DataFrame with optimized assignments
        """
        n_agencies = len(agency_ids)
        
        if priorities is None:
            priorities = np.ones(n_agencies)
            
        # Aseguramos que los vectores sean 1D y de tipo float64
        c = (-1.0 * np.asarray(priorities)).ravel().astype(np.float64)
        A_ub = np.ones((1, n_agencies), dtype=np.float64)
        b_ub = np.asarray([self.capacity_limit], dtype=np.float64)
        
        # Límites por agencia (basados en los intervalos de confianza del modelo)
        # Aseguramos escalares nativos de python
        lb_flat = np.asarray(lower_bounds).ravel()
        ub_flat = np.asarray(upper_bounds).ravel()
        
        bounds = []
        for lb, ub in zip(lb_flat, ub_flat):
            l = max(0.0, float(lb))
            u = max(l, float(ub))
            bounds.append((l, u))
        
        # Intentar optimización con parámetros limpios
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            assignments = np.round(res.x).astype(int)
        else:
            # Fallback: proporcional a la predicción si el solver falla (ej. por infeasibilidad)
            print(f"⚠️ Optimizer fallback (Infeasible? sum(LB)={sum(lower_bounds):.1f} > Cap={self.capacity_limit})")
            total_pred = sum(predicted_demands)
            if total_pred > self.capacity_limit:
                ratio = self.capacity_limit / total_pred
                assignments = np.round(predicted_demands * ratio).astype(int)
            else:
                assignments = np.round(predicted_demands).astype(int)
                
        return pd.DataFrame({
            'agency_id': agency_ids,
            'predicted_demand': predicted_demands,
            'conf_lower': lower_bounds,
            'conf_upper': upper_bounds,
            'optimized_assignment': assignments
        })

if __name__ == "__main__":
    # Test rápido
    opt = SeatOptimizer(capacity_limit=100)
    agencies = ["A1", "A2", "A3"]
    demands = [40, 50, 60] # Total 150 > 100
    lowers = [30, 40, 50]
    uppers = [50, 70, 80]
    ratings = [0.9, 0.7, 0.5] # A1 es la mejor agencia
    
    result = opt.optimize_allocation(agencies, demands, lowers, uppers, ratings)
    print("Test de Optimización (Capacidad=100):")
    print(result)
    print("Total Asignado:", result['optimized_assignment'].sum())
