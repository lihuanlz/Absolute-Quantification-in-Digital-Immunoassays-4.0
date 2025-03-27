


import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import BoundaryNorm

B0 = 1e-18  

K1 = 1/(1e-9)
K2 = 1/(1e-9)  
K3 = (5.1e6)/(5.1e-9)  


def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solution exists")
    
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    
    
    return max(0, min(root1, root2))


A0_range = np.logspace(-12, -7, 100)
C0_range = np.logspace(-12, -7, 100)
D0_values = [1e-9, 1e-12, 1e-15, 1e-16]

for D0 in D0_values:
    ABCD_concentration = np.zeros((len(A0_range), len(C0_range)))
    
    for i, A0 in enumerate(A0_range):
        for j, C0 in enumerate(C0_range):
            
            a1 = K1
            b1 = -(K1 * (A0 + B0) + 1)
            c1 = K1 * A0 * B0

            x1 = solve_quadratic(a1, b1, c1)

            
            a2 = K2
            b2 = -(K2 * (x1 + C0) + 1)
            c2 = K2 * x1 * C0

            y1 = solve_quadratic(a2, b2, c2)

            
            a3 = K3
            b3 = -(K3 * (y1 + D0) + 1)
            c3 = K3 * y1 * D0

            z1 = solve_quadratic(a3, b3, c3)

            
            ABCD_concentration[i, j] = z1

    
    A0_grid, C0_grid = np.meshgrid(A0_range, C0_range)
    plt.figure()
    levels = np.linspace(0, B0, 100)  
    cmap = plt.get_cmap('rainbow')
    norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N)
 
    

    
    cp = plt.contourf(A0_grid, C0_grid, ABCD_concentration.T, levels=levels, cmap=cmap, norm=norm)

    
    
    cbar = plt.colorbar(cp, ticks=levels)

    
    num_ticks = 11
    cbar.set_ticks(np.linspace(0, B0, num_ticks))
    
        
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Ab1 (M)')
    plt.ylabel('Ab2 (M)')
    plt.title(f'Complex concentration for SPA = {D0} M')

    
    specific_levels = [0.9 * B0, 0.8 * B0, 0.7 * B0, 0.6 * B0, 0.5 * B0, 0.4 * B0, 0.3 * B0, 0.2 * B0, 0.1 * B0, 0.05 * B0, 0.01 * B0]
    specific_levels.sort()  
    cs = plt.contour(A0_grid, C0_grid, ABCD_concentration.T, levels=specific_levels, colors='white', linestyles='-')
    plt.clabel(cs, fmt={level: f'{(level/B0)*100:.1f}%' for level in specific_levels}, inline=True, fontsize=8)


    
    filename = f'figure S2A_{D0}.svg'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()  
