"""
Economic Entropy Model for the Yusuf-Grondona System.
Implements entropy production, negentropy capture, and total entropy dynamics.
"""

import numpy as np
from scipy.integrate import odeint

class EntropyModel:
    """
    Economic entropy model with production and negentropy capture.
    
    State variables:
        S_total(t): Total entropy
        V_stock(t): Physical stock value
        D(t): Debt (for debt-based system)
    
    Entropy dynamics:
        dS_total/dt = S_prod - S_neg
        S_prod = S_min + eta * Q(t) + phi * r * D(t)
        S_neg = gamma * dV_stock/dt
    """
    
    def __init__(self, S_min=10.0, eta=0.5, gamma=0.8, phi=0.8,
                 Q_base=50.0, Q_amp=10.0, tau=7.0):
        """
        Args:
            S_min: Incompressible entropy floor
            eta: Dissipation coefficient
            gamma: Negentropic efficiency
            phi: Fraction of dissipated interest
            Q_base: Base production
            Q_amp: Production amplitude
            tau: Cycle period (years)
        """
        self.S_min = S_min
        self.eta = eta
        self.gamma = gamma
        self.phi = phi
        self.Q_base = Q_base
        self.Q_amp = Q_amp
        self.tau = tau
        
    def production(self, t):
        """
        Production with seasonal cycle.
        Q(t) = Q_base + Q_amp * sin(2*pi*t/tau)
        """
        return self.Q_base + self.Q_amp * np.sin(2 * np.pi * t / self.tau)
    
    def production_entropy(self, t, D=0.0, r=0.0):
        """
        Entropy production.
        S_prod = S_min + eta * Q(t) + phi * r * D
        """
        Q = self.production(t)
        return self.S_min + self.eta * Q + self.phi * r * D
    
    def negentropy_capture(self, dV_stock_dt):
        """
        Negentropy capture.
        S_neg = gamma * dV_stock/dt
        """
        return self.gamma * dV_stock_dt
    
    def yusuf_storage_rule(self, S, Y, C_base=50.0, Y_high=60.0, Y_low=40.0,
                           S_max=1000.0, S_min_stock=100.0, eta_stock=0.7):
        """
        Yusufian storage rule for stock dynamics.
        
        Args:
            S: Current stock
            Y: Current production
            C_base: Basic consumption need
            Y_high: Production threshold for abundance
            Y_low: Production threshold for scarcity
            S_max: Maximum storage capacity
            S_min_stock: Minimum reserve
            eta_stock: Storage fraction in abundance
        
        Returns:
            dS/dt: Stock change rate
        """
        if Y > Y_high and S < S_max:
            # Abundance: store fraction of surplus
            surplus = Y - C_base
            dS = eta_stock * surplus
            dS = min(dS, S_max - S)
        elif Y < Y_low and S > S_min_stock:
            # Scarcity: drawdown
            deficit = C_base - Y
            dS = -deficit
            dS = max(dS, S_min_stock - S)
        else:
            dS = 0.0
        return dS
    
    def simulate_yusuf(self, S0=500.0, V0=500.0, T=100.0, dt=0.01,
                       C_base=50.0, Y_high=60.0, Y_low=40.0,
                       S_max=1000.0, S_min_stock=100.0, eta_stock=0.7):
        """
        Simulate the Yusuf-Grondona system (zero debt).
        
        Returns:
            Dictionary with results
        """
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize arrays
        S_stock = np.zeros(n_steps)
        V_stock = np.zeros(n_steps)
        S_total = np.zeros(n_steps)
        S_prod = np.zeros(n_steps)
        S_neg = np.zeros(n_steps)
        Q = np.zeros(n_steps)
        Y = np.zeros(n_steps)
        Lambda = np.zeros(n_steps)
        
        # Initial conditions
        S_stock[0] = S0
        V_stock[0] = V0
        S_total[0] = self.S_min
        
        for i in range(n_steps - 1):
            # Production
            Q[i] = self.production(t[i])
            
            # Yusufian production cycle (for storage rule)
            Y[i] = 50.0 + 20.0 * np.sin(2 * np.pi * t[i] / 14)
            
            # Stock dynamics (Yusufian rule)
            dS_stock = self.yusuf_storage_rule(
                S_stock[i], Y[i], C_base, Y_high, Y_low,
                S_max, S_min_stock, eta_stock
            )
            S_stock[i+1] = S_stock[i] + dS_stock * dt
            S_stock[i+1] = np.clip(S_stock[i+1], S_min_stock, S_max)
            
            # Stock value (proportional to stock)
            V_stock[i+1] = V_stock[i] + dS_stock * dt
            
            # Entropy production (zero debt)
            S_prod[i] = self.production_entropy(t[i], D=0.0, r=0.0)
            
            # Negentropy capture
            dV = (V_stock[i+1] - V_stock[i]) / dt
            S_neg[i] = self.negentropy_capture(dV)
            
            # Total entropy
            S_total[i+1] = S_total[i] + (S_prod[i] - S_neg[i]) * dt
            
            # Lambda (zero debt)
            Lambda[i] = 0.0
        
        # Last values
        Q[-1] = self.production(t[-1])
        Y[-1] = 50.0 + 20.0 * np.sin(2 * np.pi * t[-1] / 14)
        S_prod[-1] = self.production_entropy(t[-1], D=0.0, r=0.0)
        Lambda[-1] = 0.0
        
        return {
            't': t,
            'S_stock': S_stock,
            'V_stock': V_stock,
            'S_total': S_total,
            'S_prod': S_prod,
            'S_neg': S_neg,
            'Q': Q,
            'Y': Y,
            'Lambda': Lambda,
            'system': 'yusuf'
        }
    
    def simulate_debt(self, D0=100.0, r=0.05, V0=500.0, T=100.0, dt=0.01,
                      C_base=50.0, Y_high=60.0, Y_low=40.0,
                      S_max=1000.0, S_min_stock=100.0, eta_stock=0.7):
        """
        Simulate the debt-based system.
        
        Returns:
            Dictionary with results
        """
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize arrays
        S_stock = np.zeros(n_steps)
        V_stock = np.zeros(n_steps)
        S_total = np.zeros(n_steps)
        S_prod = np.zeros(n_steps)
        S_neg = np.zeros(n_steps)
        Q = np.zeros(n_steps)
        Y = np.zeros(n_steps)
        D = np.zeros(n_steps)
        Lambda = np.zeros(n_steps)
        
        # Initial conditions
        S_stock[0] = 500.0
        V_stock[0] = V0
        S_total[0] = self.S_min
        D[0] = D0
        
        for i in range(n_steps - 1):
            # Production
            Q[i] = self.production(t[i])
            
            # Debt dynamics
            D[i+1] = D[i] * (1 + r * dt)
            
            # Yusufian production cycle (for storage rule)
            Y[i] = 50.0 + 20.0 * np.sin(2 * np.pi * t[i] / 14)
            
            # Stock dynamics (Yusufian rule with debt service)
            dS_stock = self.yusuf_storage_rule(
                S_stock[i], Y[i], C_base, Y_high, Y_low,
                S_max, S_min_stock, eta_stock
            )
            # Debt service reduces stock
            dS_stock -= self.phi * r * D[i]
            S_stock[i+1] = S_stock[i] + dS_stock * dt
            S_stock[i+1] = np.clip(S_stock[i+1], 0, S_max)
            
            # Stock value
            V_stock[i+1] = max(0, V_stock[i] + dS_stock * dt)
            
            # Entropy production (with debt)
            S_prod[i] = self.production_entropy(t[i], D[i], r)
            
            # Negentropy capture
            dV = (V_stock[i+1] - V_stock[i]) / dt
            S_neg[i] = self.negentropy_capture(max(0, dV))
            
            # Total entropy
            S_total[i+1] = S_total[i] + (S_prod[i] - S_neg[i]) * dt
            
            # Lambda
            Lambda[i] = (D[i] * r) / (self.S_min + self.eta * Q[i] + 1e-10)
        
        # Last values
        Q[-1] = self.production(t[-1])
        Y[-1] = 50.0 + 20.0 * np.sin(2 * np.pi * t[-1] / 14)
        S_prod[-1] = self.production_entropy(t[-1], D[-1], r)
        Lambda[-1] = (D[-1] * r) / (self.S_min + self.eta * Q[-1] + 1e-10)
        
        return {
            't': t,
            'S_stock': S_stock,
            'V_stock': V_stock,
            'S_total': S_total,
            'S_prod': S_prod,
            'S_neg': S_neg,
            'Q': Q,
            'Y': Y,
            'D': D,
            'Lambda': Lambda,
            'system': 'debt'
        }
