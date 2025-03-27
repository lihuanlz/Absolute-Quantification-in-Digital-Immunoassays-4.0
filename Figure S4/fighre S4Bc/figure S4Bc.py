

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm  

N = 2000000
Ms = 100000
error_limit = 0.11

N_values = np.linspace(1, N, num=100, dtype=int)  
M_N_ratios = np.logspace(np.log10(1/N), np.log10(int(-np.log(1/N))), num=100)



plt.figure(figsize=(10, 8))




data_list = []

for N in N_values:
    for M_N_ratio in M_N_ratios:
        M = M_N_ratio * N  
        if M > 0 and (Ms/N) > 0:  
            C1 = (1 - np.exp(-M / N)) / (1 - np.exp(-Ms / N))
            C2 = M / Ms
            relative_error = np.abs((C1 - C2) / C2)
            
            if relative_error < error_limit:
                data_list.append({'N': N, 'M_N_ratio': M_N_ratio, 'relative_error': relative_error})

df = pd.DataFrame(data_list)

min_error = df['relative_error'].min()
max_error = df['relative_error'].max()
bounds = np.linspace(min_error, max_error, 11)  
norm = BoundaryNorm(boundaries=bounds, ncolors=10)  


scatter = plt.scatter(df['M_N_ratio'], df['N'], c=df['relative_error'], cmap='tab10', norm=norm, s=10, alpha=1)


cbar = plt.colorbar(scatter)
cbar.set_label('Relative Error', fontsize=15)
cbar.ax.tick_params(labelsize=15)  

cbar.set_ticks(bounds)  
cbar.set_ticklabels(["{:.2f}".format(b) for b in bounds])  


grid_df = df.pivot_table(index='N', columns='M_N_ratio', values='relative_error', aggfunc='min')
X, Y = np.meshgrid(grid_df.columns, grid_df.index)
Z = grid_df.values


contour_lines = plt.contour(X, Y, Z, levels=[0.05, 0.1], colors=['blue', 'red'], linewidths=1)









plt.xlabel('M/N ratio', fontsize=15)
plt.ylabel('N', fontsize=15)
plt.title(f'Scatter Plot with Relative Error\nMs={Ms}', fontsize=15)
plt.xscale('log')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('figure_S7B_scatter.svg', dpi=600)
plt.show()

df.to_csv('5_6relative_error_data.csv', index=False)

