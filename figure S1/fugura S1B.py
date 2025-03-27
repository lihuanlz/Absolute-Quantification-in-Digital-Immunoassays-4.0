

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def equations(vars, q1, q2, K1, K2, K3, K4):
    x1, x2, x3, x4, x5, x6 = vars
    k1, k2, k3, k4 = K1, K2, K3, K4
    k5, k6, k7, k8 = K3, K4, K1, K2
    eq1 = k3*x2*x6 + k7*x3*x5 - (k4*x1 + k8*x1)
    eq2 = k1*x4*x5 + k4*x1 - (k2*x2 + k3*x2*x6)
    eq3 = k5*x4*x6 + k8*x1 - (k6*x3 + k7*x3*x5)
    eq4 = x1 + x2 + x3 + x4 - p
    eq5 = x1 + x2 + x5 - q1
    eq6 = x1 + x3 + x6 - q2
    return [eq1, eq2, eq3, eq4, eq5, eq6]


q1 = 1E-9  
q2 = 1E-9  
p = 1E-18  


K1_values = np.logspace(4, 6, 100)
K2_values = np.logspace(-5, -3, 100)
K3_values = np.logspace(4, 6, 3)
K4_values = np.logspace(-5, -3, 3)


x1_values = np.zeros((len(K1_values), len(K2_values), len(K3_values), len(K4_values)))


for i, K1 in enumerate(K1_values):
    for j, K2 in enumerate(K2_values):
        for m, K3 in enumerate(K3_values):
            for n, K4 in enumerate(K4_values):
                
                initial_guess = [0, 0, 0, p, q1, q2]
                
                solution = fsolve(equations, initial_guess, args=(q1, q2, K1, K2, K3, K4), xtol=1e-10, maxfev=10000)
                
                x1_values[i, j, m, n] = solution[0]


K1_log = np.log10(K1_values)
K2_log = np.log10(K2_values)


K3_labels = ['K3=%.1e' % K3 for K3 in K3_values]
K4_labels = ['K4=%.1e' % K4 for K4 in K4_values]




norm = Normalize(vmin=0, vmax=p)







levels = np.linspace(0, p, 101)  


ratios = sorted([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.40, 0.30, 0.20, 0.10])
levels_line = [p * ratio for ratio in ratios] 


fig, axes = plt.subplots(len(K3_values), len(K4_values), figsize=(18, 15), sharex=True, sharey=True)

for m, K3 in enumerate(K3_values):
    for n, K4 in enumerate(K4_values):
        ax = axes[m, n]
        
        X, Y = np.meshgrid(K1_log, K2_log)
        
        Z = x1_values[:, :, m, n].T
        
        
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cm.rainbow, vmin=0, vmax=p)
        
        
        cbar = fig.colorbar(contour, ax=ax)
    
        
        num_ticks = 11
        tick_values = np.linspace(0, p, num_ticks)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{t:.1e}' for t in tick_values])  
        
        
        CS = ax.contour(X, Y, Z, levels=levels_line, colors='white')
        
        fmt = {lvl: "{:.0f}%".format((lvl/p) * 100) for lvl in levels}
        
        ax.clabel(CS, inline=True, fontsize=8, fmt=fmt)
        
        ax.set_title(f'Kon2 (M$^{-1}$s$^{-1}$)={K3:.1e}, Koff2 (s$^{-1}$)={K4:.1e}')
        ax.set_xlabel('Kon1 (M$^{-1}$s$^{-1}$)')
        ax.set_ylabel('Koff1 (s$^{-1}$)')


dpi_value = 300
plt.tight_layout()
plt.savefig('figure S1B.svg', dpi=dpi_value)
plt.show()
