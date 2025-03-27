


import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def equations(vars, q1, q2, K1, K2):
    x1, x2, x3, x4, x5, x6 = vars
    k1, k2, k3, k4, k5, k6, k7, k8 = K1, K2, K1, K2, K1, K2, K1, K2
    
    eq1 = k3*x2*x6 + k7*x3*x5 - (k4*x1 + k8*x1)
    eq2 = k1*x4*x5 + k4*x1 - (k2*x2 + k3*x2*x6)
    eq3 = k5*x4*x6 + k8*x1 - (k6*x3 + k7*x3*x5)
    eq4 = x1 + x2 + x3 + x4 - p
    eq5 = x1 + x2 + x5 - q1
    eq6 = x1 + x3 + x6 - q2
    return [eq1, eq2, eq3, eq4, eq5, eq6]

q1_values = np.logspace(-10, -6, 500)
q2_values = np.logspace(-10, -6, 500)

K1 = 1e5
K2 = 1e-4

p = 1E-18

x1_values = np.zeros((len(q2_values), len(q1_values)))  

max_x1 = 0
max_q1 = 0
max_q2 = 0

for i, q2 in enumerate(q2_values):
    for j, q1 in enumerate(q1_values):

        initial_guess = [0, 0, 0, 0, q1, q2]

        solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(q1, q2, K1, K2), xtol=1e-5, maxfev=1000, full_output=True)
        
        if ier != 1:
            
            x1_values[i, j] = np.nan
            continue

        x1_values[i, j] = solution[0]
        
        if solution[0] > max_x1:
            max_x1 = solution[0]
            max_q1 = q1
            max_q2 = q2

print("Maximum x1 value is:", max_x1)
print("Corresponding q1 is:", max_q1)
print("Corresponding q2 is:", max_q2)

X, Y = np.meshgrid(q1_values, q2_values)

Z = x1_values


print("x1_values的最小值:", np.nanmin(Z))
print("x1_values的最大值:", np.nanmax(Z))
print("x1_values中NaN的数量:", np.isnan(Z).sum())


ratios = sorted([0.95, 0.70, 0.50, 0.30, 0.20])
levels = [max_x1 * ratio for ratio in ratios]

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)


CS = plt.contourf(X, Y, Z, levels=100, cmap='rainbow')


cbar = plt.colorbar()


num_ticks = 11
cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, num_ticks))





CS2 = plt.contour(X, Y, Z, levels=levels, colors='white')
fmt = {lvl: "{:.0f}%".format(ratio * 100) for lvl, ratio in zip(levels, ratios)}
plt.clabel(CS2, inline=True, fontsize=15, fmt=fmt)

plt.xscale('log')
plt.yscale('log')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel('Ab1 (mol/L)', fontsize=15)
plt.ylabel('Ab2 (mol/L)', fontsize=15)
plt.title('Contour plot of Complex (mol/L)', fontsize=15)
plt.tight_layout()
dpi_value = 300
plt.savefig('figure_S1A.svg', dpi=dpi_value)
plt.show()
