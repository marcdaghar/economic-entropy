"""
Visualization functions for the entropy article.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Rectangle

# Set publication-ready style
plt.style.use('seaborn-v0-8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the entropy article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_entropy_comparison(self, results_yusuf, results_debt):
        """
        Figure 1: Entropy comparison (Yusuf vs Debt-based).
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Panel 1: Entropy evolution
        ax = axes[0]
        
        # Yusuf system
        t = results_yusuf['t']
        S_total_y = results_yusuf['S_total']
        ax.plot(t, S_total_y, linewidth=2.5, color='blue', 
                label='Yusuf-Grondona (bounded)')
        
        # Debt system
        S_total_d = results_debt['S_total']
        ax.plot(t, S_total_d, linewidth=2.5, color='red', 
                label='Debt-based (divergent)')
        
        # Add entropy threshold
        S_crit = results_debt['S_total'][-1] / 2
        ax.axhline(y=S_crit, color='black', linestyle='--', linewidth=1.5,
                   label=f'$S_{{\\text{{crit}}}} = {S_crit:.1f}$')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Total entropy $S_{total}(t)$', fontsize=12)
        ax.set_title('Entropy Evolution: Yusuf-Grondona vs Debt-Based System', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Entropy production vs negentropy (Yusuf)
        ax = axes[1]
        S_prod = results_yusuf['S_prod']
        S_neg = results_yusuf['S_neg']
        ax.plot(t, S_prod, linewidth=2, color='green', 
                label=r'$S_{prod}(t)$ (production)')
        ax.plot(t, S_neg, linewidth=2, color='purple', 
                label=r'$S_{neg}(t)$ (negentropy)')
        ax.fill_between(t, S_prod, S_neg, where=(S_prod > S_neg),
                        color='red', alpha=0.2, label='Net entropy production')
        ax.fill_between(t, S_prod, S_neg, where=(S_prod <= S_neg),
                        color='blue', alpha=0.2, label='Net negentropy capture')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Entropy rate', fontsize=12)
        ax.set_title('Entropy Balance in the Yusuf-Grondona System', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/entropy_comparison.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/entropy_comparison.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_entropy_phase_portrait(self, results_yusuf, results_debt):
        """
        Figure 2: Phase portrait of entropy dynamics.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Yusuf system
        ax = axes[0]
        S_total = results_yusuf['S_total']
        dS_total = np.gradient(S_total, results_yusuf['t'])
        ax.plot(S_total, dS_total, linewidth=2, color='blue', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.scatter(S_total[0], dS_total[0], color='green', s=100, 
                   label='Initial', marker='o')
        ax.scatter(S_total[-1], dS_total[-1], color='red', s=100,
                   label='Final', marker='s')
        
        # Add limit cycle indication
        ax.annotate('Limit cycle', xy=(S_total[-100], 0), 
                   xytext=(S_total[-100] + 50, 50),
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel(r'$S_{total}$', fontsize=12)
        ax.set_ylabel(r'$dS_{total}/dt$', fontsize=12)
        ax.set_title('Yusuf-Grondona: Converges to Limit Cycle', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Debt system
        ax = axes[1]
        S_total = results_debt['S_total']
        dS_total = np.gradient(S_total, results_debt['t'])
        ax.plot(S_total, dS_total, linewidth=2, color='red', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.scatter(S_total[0], dS_total[0], color='green', s=100, 
                   label='Initial', marker='o')
        ax.scatter(S_total[-1], dS_total[-1], color='red', s=100,
                   label='Final', marker='s')
        
        # Add divergence indication
        ax.annotate('Diverges to infinity', xy=(S_total[-1], dS_total[-1]), 
                   xytext=(S_total[-1] - 200, dS_total[-1] - 100),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel(r'$S_{total}$', fontsize=12)
        ax.set_ylabel(r'$dS_{total}/dt$', fontsize=12)
        ax.set_title('Debt-Based: Diverges to Infinity', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/entropy_phase_portrait.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/entropy_phase_portrait.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_entropy_lambda(self, results_debt):
        """
        Figure 3: Relationship between Lambda and entropy.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        Lambda = results_debt['Lambda']
        S_total = results_debt['S_total']
        t = results_debt['t']
        D = results_debt['D']
        Q = results_debt['Q']
        
        # Panel 1: Lambda and Entropy over time
        ax = axes[0]
        ax.plot(t, Lambda, linewidth=2.5, color='darkred', label=r'$\Lambda(t)$')
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
                   label=r'$\Lambda = 1$ (critical threshold)')
        
        ax2 = ax.twinx()
        ax2.plot(t, S_total, linewidth=2, color='blue', linestyle=':', 
                 label=r'$S_{total}(t)$')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'$\Lambda(t)$', fontsize=12, color='darkred')
        ax.tick_params(axis='y', labelcolor='darkred')
        ax2.set_ylabel(r'$S_{total}(t)$', fontsize=12, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax.set_title(r'$\Lambda$ and Entropy Evolution', fontsize=14)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Lambda vs Entropy (phase portrait)
        ax = axes[1]
        ax.plot(S_total, Lambda, linewidth=2, color='purple', alpha=0.8)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                   label=r'$\Lambda = 1$')
        ax.axvline(x=S_total[np.where(Lambda >= 1.0)[0][0]] if np.any(Lambda >= 1.0) else 0,
                   color='red', linestyle=':', linewidth=2,
                   label=r'$S_{crit}$')
        
        # Annotate critical point
        if np.any(Lambda >= 1.0):
            idx = np.where(Lambda >= 1.0)[0][0]
            ax.scatter(S_total[idx], Lambda[idx], color='red', s=150, zorder=5)
            ax.annotate(f'Critical point\n($\Lambda=1, S={S_total[idx]:.1f}$)',
                       xy=(S_total[idx], Lambda[idx]),
                       xytext=(S_total[idx] + 50, Lambda[idx] + 1),
                       arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel(r'$S_{total}$', fontsize=12)
        ax.set_ylabel(r'$\Lambda$', fontsize=12)
        ax.set_title(r'Phase Portrait: $\Lambda$ vs $S_{total}$', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/entropy_lambda.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/entropy_lambda.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_entropy_stock_relationship(self, results_yusuf, results_debt):
        """
        Figure 4: Relationship between entropy and stock.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Yusuf system
        ax = axes[0]
        S_stock = results_yusuf['S_stock']
        S_total = results_yusuf['S_total']
        ax.plot(S_stock, S_total, linewidth=2, color='blue', alpha=0.7)
        ax.scatter(S_stock[0], S_total[0], color='green', s=100, 
                   label='Initial', marker='o')
        ax.scatter(S_stock[-1], S_total[-1], color='red', s=100,
                   label='Final', marker='s')
        ax.set_xlabel(r'Stock $S(t)$', fontsize=12)
        ax.set_ylabel(r'$S_{total}$', fontsize=12)
        ax.set_title('Yusuf-Grondona: Entropy vs Stock', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Debt system
        ax = axes[1]
        S_stock = results_debt['S_stock']
        S_total = results_debt['S_total']
        ax.plot(S_stock, S_total, linewidth=2, color='red', alpha=0.7)
        ax.scatter(S_stock[0], S_total[0], color='green', s=100, 
                   label='Initial', marker='o')
        ax.scatter(S_stock[-1], S_total[-1], color='red', s=100,
                   label='Final', marker='s')
        ax.set_xlabel(r'Stock $S(t)$', fontsize=12)
        ax.set_ylabel(r'$S_{total}$', fontsize=12)
        ax.set_title('Debt-Based: Entropy vs Stock (Divergence)', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/entropy_stock_relationship.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/entropy_stock_relationship.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, results_yusuf, results_debt):
        """
        Generate all figures for the article.
        """
        print("Generating Figure 1: Entropy comparison...")
        self.figure_entropy_comparison(results_yusuf, results_debt)
        
        print("Generating Figure 2: Phase portrait...")
        self.figure_entropy_phase_portrait(results_yusuf, results_debt)
        
        print("Generating Figure 3: Lambda vs Entropy...")
        self.figure_entropy_lambda(results_debt)
        
        print("Generating Figure 4: Entropy vs Stock...")
        self.figure_entropy_stock_relationship(results_yusuf, results_debt)
        
        print("All figures generated successfully!")
