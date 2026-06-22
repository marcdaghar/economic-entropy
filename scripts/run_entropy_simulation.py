#!/usr/bin/env python3
"""
Run entropy simulations for Yusuf-Grondona and debt-based systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.entropy_model import EntropyModel
from src.visualization import FigureGenerator

def run_yusuf_simulation():
    """
    Run Yusuf-Grondona entropy simulation.
    """
    print("=" * 60)
    print("Running Yusuf-Grondona Entropy Simulation...")
    print("=" * 60)
    
    model = EntropyModel(
        S_min=10.0,
        eta=0.5,
        gamma=0.8,
        phi=0.8,
        Q_base=50.0,
        Q_amp=10.0,
        tau=7.0
    )
    
    results = model.simulate_yusuf(
        S0=500.0,
        V0=500.0,
        T=100.0,
        dt=0.01,
        C_base=50.0,
        Y_high=60.0,
        Y_low=40.0,
        S_max=1000.0,
        S_min_stock=100.0,
        eta_stock=0.7
    )
    
    print(f"\nYusuf-Grondona Results:")
    print(f"  Final entropy: {results['S_total'][-1]:.2f}")
    print(f"  Entropy range: [{np.min(results['S_total']):.2f}, {np.max(results['S_total']):.2f}]")
    print(f"  Final stock: {results['S_stock'][-1]:.2f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/results_yusuf.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_debt_simulation():
    """
    Run debt-based entropy simulation.
    """
    print("\n" + "=" * 60)
    print("Running Debt-Based Entropy Simulation...")
    print("=" * 60)
    
    model = EntropyModel(
        S_min=10.0,
        eta=0.5,
        gamma=0.8,
        phi=0.8,
        Q_base=50.0,
        Q_amp=10.0,
        tau=7.0
    )
    
    results = model.simulate_debt(
        D0=100.0,
        r=0.05,
        V0=500.0,
        T=100.0,
        dt=0.01,
        C_base=50.0,
        Y_high=60.0,
        Y_low=40.0,
        S_max=1000.0,
        S_min_stock=100.0,
        eta_stock=0.7
    )
    
    print(f"\nDebt-Based Results:")
    print(f"  Final entropy: {results['S_total'][-1]:.2f}")
    print(f"  Final debt: {results['D'][-1]:.2f}")
    print(f"  Final Lambda: {results['Lambda'][-1]:.2f}")
    
    # Save results
    with open('data/results_debt.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_parameter_sweep():
    """
    Run parameter sweep for sensitivity analysis.
    """
    print("\n" + "=" * 60)
    print("Running Parameter Sweep...")
    print("=" * 60)
    
    r_values = np.linspace(0.01, 0.10, 10)
    D0_values = np.linspace(50, 200, 10)
    
    results_sweep = []
    
    for r in r_values:
        for D0 in D0_values:
            model = EntropyModel()
            results = model.simulate_debt(
                D0=D0,
                r=r,
                V0=500.0,
                T=50.0,
                dt=0.01
            )
            results_sweep.append({
                'r': r,
                'D0': D0,
                'final_entropy': results['S_total'][-1],
                'final_lambda': results['Lambda'][-1],
                'crossing_time': results['t'][np.where(results['Lambda'] >= 1.0)[0][0]] 
                                 if np.any(results['Lambda'] >= 1.0) else np.inf
            })
    
    # Save sweep results
    with open('data/results_sweep.pkl', 'wb') as f:
        pickle.dump(results_sweep, f)
    
    print(f"\nSweep complete: {len(results_sweep)} runs")
    
    return results_sweep

def main():
    """
    Main execution function.
    """
    # Run simulations
    results_yusuf = run_yusuf_simulation()
    results_debt = run_debt_simulation()
    results_sweep = run_parameter_sweep()
    
    print("\n" + "=" * 60)
    print("All simulations complete. Results saved to data/ directory.")
    print("=" * 60)
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_yusuf, results_debt)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
